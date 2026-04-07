# This is a sample Python script.
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
import os


'''
LAST UPDATED 11/10/2021, lsdr
'''

## Process images in parallel

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file

OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep


os.chdir(OR_PATH) # Come back to the directory where the code resides , all files will be left on this directory

n_epoch = 15
BATCH_SIZE = 30
LR = 0.001
EARLY_STOP_PATIENCE = 3
MIXUP_PROB = 0.30
MIXUP_ALPHA = 0.20

## Image processing
CHANNELS = 3
IMAGE_SIZE = 320
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NICKNAME = "Ersa"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
SAVE_MODEL = True
POS_WEIGHT = None
THRESHOLD_FILE = "threshold_decision.txt"
CLASS_SCARCITY = None


#---- Define the model ---- #

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 128, (3, 3))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(128, OUTPUTS_a)
        self.act = torch.relu

    def forward(self, x):
        x = self.pad1(self.convnorm1(self.act(self.conv1(x))))
        x = self.act(self.conv2(self.act(x)))
        return self.linear(self.global_avg_pool(x).view(-1, 128))

class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, list_IDs, type_data, target_type):
        #Initialization'
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.target_type = target_type

    def __len__(self):
        #Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        #Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label

        if self.type_data == 'train':
            y = xdf_dset_train.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")
        elif self.type_data == 'val':
            y = xdf_dset_val.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")
        else:
            y = xdf_dset_eval.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")


        if self.target_type == 2:
            labels_ohe = [ int(e) for e in y]
        else:
            labels_ohe = np.zeros(OUTPUTS_a)

            for idx, label in enumerate(range(OUTPUTS_a)):
                if label == y:
                    labels_ohe[idx] = 1

        y = torch.FloatTensor(labels_ohe)

        if self.type_data == 'train':
            file = DATA_DIR + xdf_dset_train.id.get(ID)
        elif self.type_data == 'val':
            file = DATA_DIR + xdf_dset_val.id.get(ID)
        else:
            file = DATA_DIR + xdf_dset_eval.id.get(ID)

        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0

        if self.type_data == 'train':
            img = self._scarcity_aware_augment(img, labels_ohe)

        img = np.repeat(img[:, :, None], 3, axis=2)
        img = (img - IMG_MEAN) / IMG_STD

        X = torch.from_numpy(img).permute(2, 0, 1).float()

        return X, y

    def _scarcity_aware_augment(self, img, labels_ohe):
        if CLASS_SCARCITY is None:
            return img

        positives = [idx for idx, v in enumerate(labels_ohe) if int(v) == 1]
        if len(positives) == 0:
            sample_scarcity = 0.0
        else:
            sample_scarcity = max([float(CLASS_SCARCITY[idx]) for idx in positives])

        sample_scarcity = float(np.clip(sample_scarcity, 0.0, 1.0))
        # Augment all classes regularly; only mildly increase for scarce positives.
        p_aug = 0.55 + 0.20 * sample_scarcity
        if np.random.rand() > p_aug:
            return img

        strength = 0.60 + 0.40 * sample_scarcity
        h, w = img.shape[:2]

        ops = ["affine", "tone", "noise_blur", "cutout"]
        random.shuffle(ops)
        n_ops = 2 if np.random.rand() < 0.70 else 3

        for op in ops[:n_ops]:
            if op == "affine":
                max_angle = 8.0 + 8.0 * strength
                angle = np.random.uniform(-max_angle, max_angle)

                max_trans_frac = 0.03 + 0.05 * strength
                tx = np.random.uniform(-max_trans_frac, max_trans_frac) * w
                ty = np.random.uniform(-max_trans_frac, max_trans_frac) * h

                max_scale_jitter = 0.05 + 0.10 * strength
                scale = np.random.uniform(1.0 - max_scale_jitter, 1.0 + max_scale_jitter)

                center = (w / 2.0, h / 2.0)
                mat = cv2.getRotationMatrix2D(center, angle, scale)
                mat[0, 2] += tx
                mat[1, 2] += ty
                img = cv2.warpAffine(
                    img,
                    mat,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101
                )

            elif op == "tone":
                if np.random.rand() < 0.50:
                    gamma = np.random.uniform(0.80, 1.20 + 0.15 * strength)
                    img = np.power(np.clip(img, 0.0, 1.0), gamma)
                else:
                    bc_delta = 0.08 + 0.14 * strength
                    contrast = np.random.uniform(1.0 - bc_delta, 1.0 + bc_delta)
                    brightness = np.random.uniform(-bc_delta, bc_delta)
                    img = img * contrast + brightness

                if np.random.rand() < (0.20 + 0.20 * strength):
                    clip_limit = 1.5 + 2.0 * strength
                    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                    img = clahe.apply(img_u8).astype(np.float32) / 255.0

            elif op == "noise_blur":
                sigma = 0.006 + 0.025 * strength
                noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
                img = img + noise

                if np.random.rand() < (0.20 + 0.25 * strength):
                    ksize = 5 if np.random.rand() < 0.50 else 3
                    img = cv2.GaussianBlur(img, (ksize, ksize), 0)

            elif op == "cutout":
                if np.random.rand() < 0.70:
                    cut_h = int(h * np.random.uniform(0.08, 0.20 + 0.10 * strength))
                    cut_w = int(w * np.random.uniform(0.08, 0.20 + 0.10 * strength))
                    cut_h = max(1, min(cut_h, h))
                    cut_w = max(1, min(cut_w, w))
                    y0 = np.random.randint(0, h - cut_h + 1)
                    x0 = np.random.randint(0, w - cut_w + 1)
                    fill = float(np.random.uniform(0.0, 1.0))
                    img[y0:y0 + cut_h, x0:x0 + cut_w] = fill

        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        return img


def read_data(target_type):
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file


    ds_inputs = np.array(DATA_DIR + xdf_dset_train['id'])

    ds_targets = xdf_dset_train['target_class']

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids_train = list(xdf_dset_train.index)
    list_of_ids_val = list(xdf_dset_val.index)
    list_of_ids_eval = list(xdf_dset_eval.index)


    # Datasets
    partition = {
        'train': list_of_ids_train,
        'val' : list_of_ids_val,
        'eval': list_of_ids_eval
    }

    # Data Loaders

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True}

    training_set = Dataset(partition['train'], 'train', target_type)
    training_generator = data.DataLoader(training_set, **params)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    val_set = Dataset(partition['val'], 'val', target_type)
    val_generator = data.DataLoader(val_set, **params)

    eval_set = Dataset(partition['eval'], 'eval', target_type)
    eval_generator = data.DataLoader(eval_set, **params)

    ## Make the channel as a list to make it variable

    return training_generator, val_generator, eval_generator

def save_model(model):
    # Open the file

    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))

def model_definition(pretrained=False):
    # Define a Keras sequential model
    # Compile the model

    if pretrained == True:
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier.in_features, OUTPUTS_a)
        )
    else:
        model = CNN()

    model = model.to(device)

    if pretrained == True:
        for param in model.features.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if POS_WEIGHT is None:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0)

    save_model(model)

    return model, optimizer, criterion, scheduler

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _apply_thresholds(probs, thresholds):
    thr = np.array(thresholds, dtype=np.float32).reshape(1, -1)
    return (probs >= thr).astype(np.float32)


def _tune_thresholds(y_true, probs):
    thresholds = np.full(OUTPUTS_a, THRESHOLD, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 37)
    for c in range(OUTPUTS_a):
        best_t = THRESHOLD
        best_f1 = -1.0
        y_true_c = y_true[:, c]
        for t in grid:
            y_pred_c = (probs[:, c] >= t).astype(np.float32)
            f1_c = f1_score(y_true_c, y_pred_c, zero_division=0)
            if f1_c > best_f1:
                best_f1 = f1_c
                best_t = t
        thresholds[c] = best_t
    return thresholds


def _save_thresholds_file(thresholds):
    with open(THRESHOLD_FILE, "w") as f:
        f.write(",".join("{:.6f}".format(float(t)) for t in thresholds))

def _compute_class_scarcity_from_train_targets(targets_train):
    positives = targets_train.sum(axis=0).astype(np.float32)
    max_pos = float(np.max(positives))
    min_pos = float(np.min(positives))
    denom = max(max_pos - min_pos, 1e-6)
    scarcity = (max_pos - positives) / denom
    return np.clip(scarcity, 0.0, 1.0)


def _mixup_batch(x, y, alpha):
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    mixed_y = lam * y + (1.0 - lam) * y[index]
    return mixed_x, mixed_y


def _collect_predictions(model, ds, criterion, desc):
    pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        with tqdm(total=len(ds), desc=desc) as pbar:
            for xdata, xtarget in ds:
                xdata, xtarget = xdata.to(device), xtarget.to(device)
                output = model(xdata)
                loss = criterion(output, xtarget)
                total_loss += loss.item()
                steps += 1

                pbar.update(1)
                pbar.set_postfix_str("Loss: {:.5f}".format(total_loss / max(steps, 1)))

                pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

    return total_loss / max(steps, 1), pred_logits[1:], real_labels[1:]


def train_and_test(train_ds, val_ds, eval_ds, list_of_metrics, list_of_agg, save_on, pretrained=False):
    model, optimizer, criterion, scheduler = model_definition(pretrained)

    best_val_metric = -1.0
    best_thresholds = np.full(OUTPUTS_a, THRESHOLD, dtype=np.float32)
    no_improve_epochs = 0
    early_stop_patience = int(globals().get("EARLY_STOP_PATIENCE", 3))
    if early_stop_patience < 1:
        early_stop_patience = 1

    for epoch in range(n_epoch):
        if pretrained and epoch == 1:
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=LR * 0.2)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0)

        train_loss, steps_train = 0, 0
        model.train()

        pred_logits_train, real_labels_train = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
        with tqdm(total=len(train_ds), desc="Train Epoch {}".format(epoch)) as pbar:
            for xdata, xtarget in train_ds:
                xdata, xtarget = xdata.to(device), xtarget.to(device)
                xtarget_for_metrics = xtarget

                if MIXUP_PROB > 0.0 and MIXUP_ALPHA > 0.0 and np.random.rand() < MIXUP_PROB:
                    xdata, xtarget = _mixup_batch(xdata, xtarget, MIXUP_ALPHA)

                optimizer.zero_grad()
                output = model(xdata)
                loss = criterion(output, xtarget)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                steps_train += 1
                pbar.update(1)
                pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss / max(steps_train, 1)))

                pred_logits_train = np.vstack((pred_logits_train, output.detach().cpu().numpy()))
                real_labels_train = np.vstack((real_labels_train, xtarget_for_metrics.cpu().numpy()))

        model.eval()

        val_loss, val_logits, val_labels = _collect_predictions(
            model, val_ds, criterion, "Val Epoch {}".format(epoch)
        )
        val_probs = _sigmoid(val_logits)
        thresholds_epoch = _tune_thresholds(val_labels, val_probs)
        val_pred_labels = _apply_thresholds(val_probs, thresholds_epoch)
        val_metrics = metrics_func(list_of_metrics, list_of_agg, val_labels, val_pred_labels)
        val_metric = val_metrics.get(save_on, 0.0)
        scheduler.step(val_metric)

        train_probs = _sigmoid(pred_logits_train[1:])
        train_pred_labels = _apply_thresholds(train_probs, thresholds_epoch)
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels_train[1:], train_pred_labels)

        per_class_scores = []
        for class_idx, class_name in enumerate(class_names):
            class_f1 = f1_score(
                real_labels_train[1:, class_idx],
                train_pred_labels[:, class_idx],
                zero_division=0,
            )
            per_class_scores.append("{}={:.5f}".format(class_name, class_f1))
        print("Epoch {} Train Per-Class F1: {}".format(epoch, " | ".join(per_class_scores)))

        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in train_metrics.items():
            xstrres = xstrres + ' Train ' + met + ' {:.5f}'.format(dat)
        xstrres = xstrres + " - "
        for met, dat in val_metrics.items():
            xstrres = xstrres + ' Val ' + met + ' {:.5f}'.format(dat)
        print(xstrres)

        improved = val_metric > (best_val_metric + 1e-6)
        if improved:
            best_val_metric = val_metric
            best_thresholds = thresholds_epoch.copy()
            no_improve_epochs = 0

            if SAVE_MODEL:
                torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))
                _save_thresholds_file(best_thresholds)

                eval_loss, eval_logits, eval_labels = _collect_predictions(
                    model, eval_ds, criterion, "Eval (save) Epoch {}".format(epoch)
                )
                eval_probs = _sigmoid(eval_logits)
                eval_pred_labels = _apply_thresholds(eval_probs, best_thresholds)

                xdf_dset_results = xdf_dset_eval.copy()
                xfinal_pred_labels = []
                for i in range(len(eval_pred_labels)):
                    joined_string = ",".join(str(int(e)) for e in eval_pred_labels[i])
                    xfinal_pred_labels.append(joined_string)
                xdf_dset_results['results'] = xfinal_pred_labels
                xdf_dset_results.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)

                print("The model has been saved! Best Val {} {:.5f}".format(save_on, best_val_metric))
        else:
            no_improve_epochs += 1
            print("EarlyStopping counter: {}/{}".format(no_improve_epochs, early_stop_patience))
            if no_improve_epochs >= early_stop_patience:
                print("Early stopping triggered at epoch {}. Best Val {} {:.5f}".format(epoch, save_on, best_val_metric))
                break


def metrics_func(metrics, aggregates, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    xcont = 1
    xsum = 0
    xavg = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet =hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont +1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum/xcont
    # Ask for arguments for each metric

    return res_dict

def process_target(target_type):
    '''
        1- Binary   target = (1,0)
        2- Multiclass  target = (1...n, text1...textn)
        3- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
    :return:
    '''

    dict_target = {}
    xerror = 0

    if target_type == 2:
        ## The target comes as a string  x1, x2, x3,x4
        ## the following code creates a list
        target = np.array(xdf_data['target'].apply( lambda x : x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
        else:
            class_names = mlb.classes_
            for i in range(len(final_target)):
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
            xdf_data['target_class'] = xfinal

    if target_type == 1:
        xtarget = list(np.array(xdf_data['target'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data['target']))
        class_names=(xtarget)
        xdf_data['target_class'] = final_target

    ## We add the column to the main dataset


    return class_names


if __name__ == '__main__':

    for file in os.listdir(PATH+os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH+ os.path.sep+ "excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    ## Process Classes
    ## Input and output


    ## Processing Train dataset
    ## Target_type = 1  Multiclass   Target_type = 2 MultiLabel
    class_names = process_target(target_type = 2)

    ## Comment

    xdf_dset_all_train = xdf_data[xdf_data["split"] == 'train'].copy()
    xdf_dset_eval = xdf_data[xdf_data["split"] == 'test'].copy()

    stratify_values = xdf_dset_all_train['target_class']
    try:
        xdf_dset_train, xdf_dset_val = train_test_split(
            xdf_dset_all_train,
            test_size=0.15,
            random_state=42,
            stratify=stratify_values
        )
    except Exception:
        xdf_dset_train, xdf_dset_val = train_test_split(
            xdf_dset_all_train,
            test_size=0.15,
            random_state=42,
            stratify=None
        )

    xdf_dset_train = xdf_dset_train.copy()
    xdf_dset_val = xdf_dset_val.copy()
    xdf_dset_eval = xdf_dset_eval.copy()

    targets_train = np.array([list(map(int, str(x).split(","))) for x in xdf_dset_train['target_class']], dtype=np.float32)
    positives = targets_train.sum(axis=0)
    negatives = targets_train.shape[0] - positives
    pos_weight_np = negatives / np.maximum(positives, 1.0)
    POS_WEIGHT = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
    CLASS_SCARCITY = _compute_class_scarcity_from_train_targets(targets_train)

    ## read_data creates the dataloaders, take target_type = 2

    train_ds, val_ds, eval_ds = read_data(target_type = 2)

    OUTPUTS_a = len(class_names)

    list_of_metrics = ['f1_macro']
    list_of_agg = ['avg']

    train_and_test(train_ds, val_ds, eval_ds, list_of_metrics, list_of_agg, save_on='f1_macro', pretrained=True)
