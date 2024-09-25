# ===============================================
# Imports

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import random
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, precision_recall_curve, confusion_matrix, cohen_kappa_score

# Other Uveitis scripts used
import sys
sys.path.insert(1, os.path.dirname(__file__)+'/../Data/Metrics')
from Stats import matrix_from_predictions 
from OCI import OCI
from table_error import json_error

SOIN_save = "/data/soin/vasculitis/transformers/trained_models"

# ===============================================
# Seed fixed

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===============================================
# Losses definition

bce_loss = torch.nn.BCELoss()
ce_loss = torch.nn.CrossEntropyLoss()
occ_loss = torch.nn.CrossEntropyLoss(reduction='none')

def OCC(output, label):
    argmax_true = torch.argmax(label, dim=1)
    argmax_pred = torch.argmax(output, dim=1)
    weights = (1 + torch.abs(argmax_true - argmax_pred) / (output.size(1) - 1)).float()
    ce_loss = occ_loss(output, argmax_true)
    return (weights * ce_loss).mean()

lambda_ = 0.1
delta = 0.05
def CO(output, label):
    co_loss = ce_loss(output, label)
    kn = torch.argmax(label, dim=1)
    for k in range(label.size(1)-1):
        if k > kn:
            co_loss += lambda_*max(0, delta+output[0][k+1]-output[0][k])
        if k < kn:
            co_loss += lambda_*max(0, delta+output[0][k]-output[0][k+1])
    return co_loss

main_items = ['OD', 'ME', 'VL', 'CL']
main_index = [[0, 4], [4, 9], [9, 13], [13, 16]]

def combine_loss(loss):
    return lambda output, label: sum(loss(output[:, i[0]:i[1]], label[:, i[0]:i[1]]) for i in main_index)

def combine_bce(output, label):
    return sum(bce_loss(torch.sigmoid(output[:, j]), label[:, j]) for j in range(len(main_index)))
    
loss_functions = {'bce': lambda output, label: bce_loss(torch.sigmoid(output), label),
                  'ce': ce_loss, 'occ': OCC, 'co': CO, 'combine_bce': combine_bce}


class Trainer:

    # ===============================================
    # Initialise names and load model
    def __init__(self, data, configs):

        self.data = data
        self.configs = configs
        self.n_out = configs['n_out']
        self.item = configs['item']
        
        # Multiple frame input and other item co-prediction 
        self.best_metrics = {metric: [0, 0, 0, 0] if self.item.startswith('everything') else 0 
                             for metric in configs['metrics']}
            
        # Prints after epoch
        self.to_print = ['accuracy_train', 'confusion_matrix_train']+list(self.best_metrics)+['confusion_matrix_test']
        if self.item.startswith('everything'):
            self.to_print += ['blank']
            self.to_print = ['blank' if thing=='blank' else f'{thing}_{item}' 
                             for item in main_items for thing in self.to_print]
        self.to_print = [f'{set_name}_loss' for set_name in data]+self.to_print
        self.to_print *= configs["print_epoch"]
        self.compute_all_f1 = any('f1' in metric for metric in self.best_metrics if metric!='f1-binary_test')
        
        # Device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        torch.cuda.empty_cache()

        # Loss
        self.weighted_loss = configs["loss"].startswith('w')
        if self.weighted_loss:
            if 'everything' in self.item:
                raise ValueError("Cannot compute balanced loss for combine items")
            configs["loss"] = configs["loss"].split('_')[1]
            N_tot = len(self.data['train'])
            if self.n_out == 1:
                nones = [label for (_, label) in self.data['train']].count('none')
                self.loss_weights = torch.tensor([N_tot/nones, N_tot/(N_tot-nones)])
            else:
                labels = [np.argmax(label) for (_, label) in self.data['train']]
                self.loss_weights = torch.tensor(compute_class_weight('balanced', y=labels, 
                                                classes=np.array([x for x in range(self.n_out)])))

        # Binary loss
        if (self.n_out==1 or self.item=='everything (any)') and configs["loss"]!='bce':
            raise ValueError("Need Binary Cross-Entropy loss for binary prediction")
        configs["loss"] = 'combine_bce' if self.item=='everything (any)' else configs["loss"]
        self.loss_fn = loss_functions[configs["loss"]]
        self.loss_fn = combine_loss(self.loss_fn) if self.item=='everything' else self.loss_fn
        
    # ===============================================
    # !!! to fill in subclass !!! 
    # Load/Apply model

    def load_model(self):
        raise NotImplementedError

    def apply_model(self):
        raise NotImplementedError

    # ===============================================
    # Run experiment: training, evaluation, plots, save weights 
    def run(self):

        self.load_model()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.configs['lr']))
        print(f"model: {self.configs['model']}, device: {self.device}")
        
        for e in trange(1, self.configs['epochs']+1, leave=True, desc='epochs'):
            
            # Stop when overfitting
            if e>1 and self.metrics['accuracy_train']>self.configs["accuracy_overfit"]:
                break

            # Train model (threshold for binary classification)
            self.metrics = {'blank': '', 'thresh': [0.5, 0.5, 0.5, 0.5] if self.item=='everything (any)' else 0.5}
            self.train()

            # Evaluate model
            self.model.eval()
            if 'validation' in self.data:
                self.evaluate('validation')
            labels, preds, scores, paths = self.evaluate('test')
                    
            # Print metrics for epoch
            print('=========================   epoch:', e)
            for metric in self.to_print:
                x = self.metrics[metric]
                print(f'{metric}: {x:.4f}' if isinstance(x, float) else x)

            # Save model/plot if deserved
            self.save_check(labels, preds, scores, paths)

    # ===============================================
    # Train model, compute training metrics
    def train(self):
                
        m_loss, scores, labels, preds  = [], [], [], []
        batch_counter, loss = 0, 0
        self.model.train()
        
        for i, (path, label) in enumerate(tqdm(self.data['train'], leave=False, position=1, desc='train')):

            if batch_counter == 0:
                self.optimizer.zero_grad()
                
            # Apply model and store results
            label, loss_, pred, score = self.predict(path, label)
            loss_ = loss_*self.loss_weights[label] if self.weighted_loss else loss_
            loss += loss_
            labels.append(label)
            scores.append(score)
            preds.append(pred)

            # Train model
            if batch_counter==self.configs['batch']-1 or i==len(self.data['train'])-1:
                loss /= self.configs['batch']
                loss.backward()
                self.optimizer.step()
                m_loss.append(loss.item())
                batch_counter = -1
                loss = 0
                torch.cuda.empty_cache()
            batch_counter += 1
        
        # Compute metrics
        self.metrics['train_loss'] = np.mean(m_loss)
        self.metrics['learning_rate'] = float(self.configs['lr'])
        if self.n_out>1 and self.item!='everything (any)':
            self.multi_metrics('train', labels, preds)
        else:
            self.binary_metrics('train', labels, preds, scores)                  

    # ===============================================
    # Evaluate model on validation or test set
    def evaluate(self, set_name):
        
        with torch.no_grad():
            
            m_loss = []
            preds, scores, labels, paths  = [], [], [], []

            # Initialise for sum likelihood prediction
            if self.multi:
                self.d_likely = {os.path.dirname(path): np.zeros(self.n_out) for (path, _) in self.data[set_name]}
            
            for (path, label) in tqdm(self.data[set_name], leave=False, position=1, desc=set_name):
                
                # Apply model and store results
                label, loss, pred, score = self.predict(path, label, likely=self.multi)
                m_loss.append(loss.item())
                preds.append(pred), scores.append(score), labels.append(label)

                # Store large prediction errors
                if self.n_out>1 and self.item!='everything (any)' and score>1:
                    paths.append((path, label, pred))
                
            # Compute metrics
            self.metrics[f'{set_name}_loss'] = sum(m_loss)/len(m_loss)
            if self.n_out>1 and self.item!='everything (any)':
                self.multi_metrics(set_name, labels, preds)
            else:
                self.binary_metrics(set_name, labels, preds, scores)
        
        return labels, preds, scores, paths
    
    # ===============================================
    # Update best_metrics and save model/plot if deserved 
    def save_check(self, labels, preds, scores, paths):

        _metric = self.configs['metric_select']
            
        # Only update best_metrics when all items predicted
        if self.item.startswith('everything'):
            for k, item in enumerate(main_items):
                if self.metrics[_metric+'_'+item] > self.best_metrics[_metric][k]:
                    for metric in self.best_metrics:
                        self.best_metrics[metric][k] = self.metrics[metric+'_'+item]

        else:
            # Check if model has improved its current best performance
            if self.metrics[_metric] > self.best_metrics[_metric]:
                
                # Update every metric for this run
                for metric in self.best_metrics:
                    self.best_metrics[metric] = self.metrics[metric]

                # Save if model has improved overall best performance
                if self.metrics[_metric] > self.configs['metrics'][_metric]:
                    if self.n_out > 1:
                        self.save(labels, preds, paths)
                    else:
                        self.ROC_bootstrap(labels, scores)

    # ===============================================
    # Plot confusion matrix, save model and fill json errors file
    def save(self, labels, preds, paths):

        # Save model
        model_save = f'{SOIN_save}/{self.configs["test_set"]}'
        if not os.path.exists(model_save):
            os.makedirs(model_save)
        model_save += f'/{self.configs["model"]}__{self.item}'
        print(f'Saving weigths: {model_save} ...')
        if self.save_torch:
            torch.save(self.model.state_dict(), model_save+'.pth')
        else:
            self.model.save_pretrained(model_save)
            self.model.config.save_pretrained(model_save)

        # Save predictions
        results_save = os.path.dirname(__file__)+'/results' 
        np.save(f'{results_save}/predictions/{self.configs["test_set"]}_{self.item}_preds', preds)
        np.save(f'{results_save}/predictions/{self.configs["test_set"]}_{self.item}_labels', labels)

        # Plot and save confusion matrix
        save_folder = f'{results_save}/confusion_matrix/{self.configs["test_set"]}'
        matrix_from_predictions(labels, preds, self.item, save_folder)

        # Update json error
        json_error(self.item, paths, self.configs["test_set"])

        print('Model, plots, and json_error updated')
    
    # ===============================================
    # Apply model on an image and compute loss
    def predict(self, path, label, likely=False):

        output = self.apply_model(path)

        # Compute loss
        label = torch.tensor([label]).to(self.device).view(1, self.n_out)
        loss = self.loss_fn(output, label.float())

        # Compute prediction
        if self.n_out>1:
            if self.item == 'everything':
                label = [int(torch.argmax(label[:, i[0]:i[1]])) for i in main_index]
                pred = [int(torch.argmax(output[:, i[0]:i[1]])) for i in main_index]
                score = 0
            elif self.item == 'everything (any)':
                label = [label[:, j].float() for j in range(len(main_index))]
                pred = [(output[:, j]>self.metrics['thresh'][j]).float() for j in range(len(main_index))]
                score = [torch.sigmoid(output[:, j]) for j in range(len(main_index))]
            else:
                label, pred = int(torch.argmax(label)), int(torch.argmax(output))
                score = abs(label-pred)
            if likely:
                self.d_likely[os.path.dirname(path)] += output.view(self.n_out).cpu().numpy()
        else:
            pred = (output>self.metrics['thresh']).float()
            score = torch.sigmoid(output)

        return label, loss, pred, score
    
    # Save fpr and tpr to plot ROC curve, with confidence interval computed by bootstrap
    def ROC_bootstrap(self, labels, scores, n_bootstraps=1000, alpha=0.95):

        [labels, scores] = self.tensor_to_array([labels, scores])
        bootstrapped = []
        for _ in range(n_bootstraps):
            indices = np.random.randint(0, len(labels), len(labels))
            fpr_i, tpr_i, _ = roc_curve([labels[int(j)] for j in indices],
                                            [scores[int(j)] for j in indices], pos_label=1)
            bootstrapped.append((auc(fpr_i, tpr_i), fpr_i, tpr_i))
        bootstrapped.sort(key=lambda x: x[0])
        d_roc = {'fpr': self.metrics['fpr_test'], 'tpr': self.metrics['tpr_test']}
        auc_low, d_roc['fpr_low'], d_roc['tpr_low'] = bootstrapped[int((1-alpha)/2*n_bootstraps)]
        auc_high, d_roc['fpr_high'], d_roc['tpr_high'] = bootstrapped[int((1+alpha)/2*n_bootstraps)]
        for axis in d_roc:
            np.save(f'results/ROC/FPR-TPR/{self.item}_{axis}.npy', d_roc[axis])
        plt.plot(d_roc['fpr'], d_roc['tpr'], color='black')
        tpr_low = np.interp(d_roc['fpr_high'], d_roc['fpr_low'], d_roc['tpr_low'])
        plt.fill_between(d_roc['fpr_high'], tpr_low, d_roc['tpr_high'], color='red',
                            where=(tpr_low <= d_roc['tpr_high']), alpha=0.2, interpolate=True)
        plt.xlabel('1-Specificity (False Positive Rate)')
        plt.ylabel('Sensitivity (True Positive Rate)')
        auc_title = f"AUC: {self.metrics['auc_test']:.3f} ({auc_low:.3f} - {auc_high:.3f})"
        plt.title(f"{auc_title}, f1-score: {self.metrics['f1-score_test']:.3f}")
        plt.savefig(f'results/ROC/{self.item}.png')
        plt.close()
        
    # ===============================================
    # Metrics for binary classification
    def binary_metrics(self, dataset, labels, preds, scores, everything=True, n=0):

        # Run for all 4 main items if predicted together
        if everything and self.item=='everything (any)':
            for j in range(len(main_index)):
                self.binary_metrics(dataset+'_'+main_items[j], [label[j] for label in labels], [pred[j] for pred in preds],
                                    [score[j] for score in scores], everything=False, n=j)
            self.metrics['accuracy_'+dataset] = np.mean([self.metrics[f'accuracy_{dataset}_{item}']
                                                        for item in main_items])
            return
        
        # Tensors to arrays conversion 
        [labels, preds, scores] = self.tensor_to_array([labels, preds, scores])

        # Compute threshold from validation set prediction
        if dataset.startswith('validation'):
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            fscore = (2 * precision * recall) / (precision + recall + 1e-6)
            thresh = float(thresholds[np.argmax(fscore)])
            if self.item == 'everything (any)':
                self.metrics['thresh'][n] = self.metrics['thresh_'+main_items[n]] = thresh
            else:
                self.metrics['thresh'] = thresh

        # Compute binary metrics
        self.metrics['accuracy_'+dataset] = accuracy_score(labels, preds)
        self.metrics['f1-score_'+dataset] = f1_score(labels, preds)
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        self.metrics['auc_'+dataset] = auc(fpr, tpr)
        self.metrics['fpr_'+dataset], self.metrics['tpr_'+dataset] = fpr, tpr
        self.metrics['confusion_matrix_'+dataset] = confusion_matrix(labels, preds)

    # ===============================================
    # Metrics for multiclass classification
    def multi_metrics(self, dataset, labels, preds, penalized=False, everything=True, n=0):

        # Run for all 4 main items if predicted together
        if everything and self.item=='everything':
            for k, i in enumerate(main_index):
                self.multi_metrics(dataset+'_'+main_items[k], [label[k] for label in labels], 
                                   [pred[k] for pred in preds], penalized=False, everything=False, n=i[1]-i[0])
            self.metrics['accuracy_'+dataset] = np.mean([self.metrics[f'accuracy_{dataset}_{item}']
                                                        for item in main_items])
            return

        # Initialise for sum likelihood prediction
        if self.multi and dataset!='train':
            d_labels = {os.path.dirname(path): int(np.argmax(label)) for (path, label) in self.data[dataset]}
            labels = [d_labels[folder] for folder in d_labels]
            preds = [np.argmax(self.d_likely[folder]) for folder in self.d_likely]

        dataset += '_penalized'*penalized
        n = self.n_out if not n else n
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = sum(max(1, abs(i-j)*penalized) for x, y in zip(labels, preds) if (x, y)==(i, j))
        
        # Compute main metrics
        if not penalized:
            self.metrics['accuracy_'+dataset] = accuracy_score(labels, preds)
            self.metrics['1-OCI_'+dataset] = 1-OCI(matrix)
            self.metrics['confusion_matrix_'+dataset] = matrix
            self.metrics['f1-binary_'+dataset] = f1_score([int(bool(x)) for x in labels], [int(bool(x)) for x in preds])
            self.metrics['kappa_'+dataset] = cohen_kappa_score(labels, preds)

        if not self.compute_all_f1:
            return
        
        # Compute f1_score if needed
        f1_scores = []
        total_tp, total_fn, total_fp = 0, 0, 0
        for i in range(len(matrix)):
            tp = matrix[i, i]
            total_tp += tp
            fn = sum(matrix[:, i]) - tp
            total_fn += fn
            fp = sum(matrix[i, :]) - tp
            total_fp += fp
            f1_scores.append(tp/(tp+(fp+fn)/2) if tp+(fp+fn)!=0 else 0)
        self.metrics['f1-micro_'+dataset] = total_tp/(total_tp+(total_fp+total_fn)/2)
        self.metrics['f1-macro_'+dataset] = np.mean([f1 for f1 in f1_scores])
        self.metrics['f1-weighted_'+dataset] = np.average([f1 for f1 in f1_scores], weights=np.sum(matrix, axis=0))

        # Redo for penalized f1_score
        if not penalized:
            self.multi_metrics(dataset, labels, preds, penalized=True)