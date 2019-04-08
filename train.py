from comet_ml import Experiment as CometExperiment
from sklearn.metrics import precision_recall_fscore_support as prfs

import torch
import torch.utils.data
import torch.optim as optim
import torch.autograd as autograd

from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, download_dataset, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics, log_patches)
from utils.inference import generate_patches, log_full_image


from polyaxon_client.tracking import Experiment

import logging

"""
Initialize Parser and define arguments
"""
parser = get_parser_with_args()
opt = parser.parse_args()


"""
Initialize experiments for polyaxon and comet.ml
"""
comet = CometExperiment('QQFXdJ5M7GZRGri7CWxwGxPDN',
                        project_name=opt.project_name,
                        auto_param_logging=False,
                        parse_args=False)
comet.log_other('status', 'started')
experiment = Experiment()
logging.basicConfig(level=logging.INFO)
comet.log_parameters(vars(opt))


"""
Set up environment: define paths, download data, and set device
"""
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))
download_dataset(opt.dataset, comet)
train_loader, val_loader = get_loaders(opt)


"""
Load Model then define other aspects of the model
"""
logging.info('LOADING Model')
model = load_model(opt, dev)

criterion = get_criterion(opt)
optimizer = optim.SGD(model.parameters(), lr=opt.lr)
# optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-2)


"""
 Set starting values
"""
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}

logging.info('STARTING training')
for epoch in range(opt.epochs):
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()

    """
    Begin Training
    """
    with comet.train():
        model.train()
        logging.info('SET model mode to train!')
        batch_iter = 0
        for batch_img1, batch_img2, labels in train_loader:
            logging.info("batch info " +
                         str(batch_iter) + " - " +
                         str(batch_iter+opt.batch_size))
            batch_iter = batch_iter+opt.batch_size

            # Set variables for training
            batch_img1 = autograd.Variable(batch_img1).to(dev)
            batch_img2 = autograd.Variable(batch_img2).to(dev)
            labels = autograd.Variable(labels).long().to(dev)

            # Zero the gradient
            optimizer.zero_grad()

            # Get model predictions, calculate loss, backprop
            cd_preds = model(batch_img1, batch_img2)
            cd_loss = criterion(cd_preds, labels)
            loss = cd_loss
            loss.backward()
            optimizer.step()
            _, cd_preds = torch.max(cd_preds, 1)

            # Calculate and log other batch metrics
            cd_corrects = (100 *
                           (cd_preds.byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (opt.patch_size**2)))

            cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                                   cd_preds.data.cpu().numpy().flatten(),
                                   average='binary',
                                   pos_label=1)

            train_metrics = set_metrics(train_metrics,
                                        cd_loss,
                                        cd_corrects,
                                        cd_train_report)

            # log the batch mean metrics
            mean_train_metrics = get_mean_metrics(train_metrics)
            comet.log_metrics(mean_train_metrics)

            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        print("EPOCH TRAIN METRICS", mean_train_metrics)

    """
    Begin Validation
    """
    with comet.validate():
        model.eval()

        first_batch = True
        for batch_img1, batch_img2, labels in val_loader:
            # Set variables for training
            batch_img1 = autograd.Variable(batch_img1).to(dev)
            batch_img2 = autograd.Variable(batch_img2).to(dev)
            labels = autograd.Variable(labels).long().to(dev)

            # Get predictions and calculate loss
            cd_preds = model(batch_img1, batch_img2)
            cd_loss = criterion(cd_preds, labels)
            _, cd_preds = torch.max(cd_preds, 1)

            # If this is the first batch, comet log the loss to gauge results
            if first_batch:
                log_patches(comet,
                            epoch,
                            batch_img1,
                            batch_img2,
                            labels,
                            cd_preds)
                first_batch = False

            # Calculate and log other batch metrics
            cd_corrects = (100 *
                           (cd_preds.byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (opt.patch_size**2)))

            cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                                 cd_preds.data.cpu().numpy().flatten(),
                                 average='binary',
                                 pos_label=1)

            val_metrics = set_metrics(val_metrics,
                                      cd_loss,
                                      cd_corrects,
                                      cd_val_report)

            # log the batch mean metrics
            mean_val_metrics = get_mean_metrics(val_metrics)
            comet.log_metrics(mean_val_metrics)

            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        print("EPOCH VALIDATION METRICS", mean_val_metrics)

        """
        Output full test image
        """
        print("STARTING FULL VALIDATION IMAGE INFERENCES", mean_val_metrics)
        # Get a list of all cities we want to log for full inference
        validation_cities = opt.validation_cities.split(',')

        # Perform inference then log results for each validation city
        for city in validation_cities:
            # get a set of patches for both dates and reconstruction metadata
            p1, p2, hs, ws, lc, lr, h, w = generate_patches(opt, city)

            out = []
            for i in range(0, p1.shape[0], opt.batch_size):
                # Take a section of patches as the batch
                b1 = torch.from_numpy(p1[i:i+opt.batch_size, :, :, :]).to(dev)
                b2 = torch.from_numpy(p2[i:i+opt.batch_size, :, :, :]).to(dev)

                # Predict results
                preds = model(b1, b2)

                # Clear batches from memory
                del b1, b2

                # Flatten prediction to only max value (change v no-change)
                _, cd_preds = torch.max(preds, 1)
                cd_preds = cd_preds.data.cpu().numpy()
                out.append(cd_preds)

            # log the full image to comet.ml
            log_full_image(out, hs, ws, lc, lr, h, w,
                           opt, city, epoch, dev, comet)

    """
    Store the weights of good epochs based on validation results
    """
    if ((mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions'])
            or
            (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls'])
            or
            (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores'])):
        # Save to comet.ml and in GCS
        torch.save(model, '/tmp/checkpoint_epoch_'+str(epoch)+'.pt')
        upload_file_path = '/tmp/checkpoint_epoch_'+str(epoch)+'.pt'
        experiment.outputs_store.upload_file(upload_file_path)
        comet.log_asset('/tmp/checkpoint_epoch_'+str(epoch)+'.pt')
        best_metrics = mean_val_metrics

    # Log all train and validation metrics
    log_train_metrics = {"train_"+k: v for k, v in mean_train_metrics.items()}
    log_val_metrics = {"validate_"+k: v for k, v in mean_val_metrics.items()}
    epoch_metrics = {'epoch': epoch, **log_train_metrics, **log_val_metrics}

    experiment.log_metrics(**epoch_metrics)

    # Set experiment to running properly (for filtering out bad runs)
    comet.log_other('status', 'running')
    comet.log_epoch_end(epoch)
comet.log_other('status', 'complete')
