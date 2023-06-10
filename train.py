import os
import yaml
import argparse
import pandas as pd

from tqdm.auto import tqdm

import torch
import evaluate
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import MarkupLMProcessor
from transformers import MarkupLMForTokenClassification
from transformers import set_seed
# from torchmetrics import Precision

import utils
import input_pipeline


# TODO: add tensorboard functionality
# TODO: add new script for running inference on eval set and obtaining metrics
# TODO: look into label alignment
# TODO: is there value in labeling other subwords in each word

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# set the seed for the hugggingface package
set_seed(42)


def run_train_loop(
    batch, model, optimizer, loss_fct, device, train_metric, label_list, config
):
    # get the inputs;
    inputs = {k: v.to(device) for k, v in batch.items()}

    if config["ablation"]["run_ablation"]:
        inputs = utils.ablation(config, inputs)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(**inputs)

    # get the logits
    logits = outputs.logits

    attention_mask = inputs["attention_mask"]
    labels = inputs["labels"]

    # Only keep active parts of the loss
    num_labels = len(label_list)
    active_loss = attention_mask.view(-1) == 1
    active_logits = logits.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        labels.view(-1),
        torch.tensor(loss_fct.ignore_index).type_as(labels),
    )
    loss = loss_fct(active_logits, active_labels)

    # loss = outputs.loss
    loss.backward()
    optimizer.step()

    print("Train Loss:", loss.item())

    predictions = outputs.logits.argmax(dim=-1)
    labels = batch["labels"]
    preds, refs = utils.convert_preds_to_labels(predictions,
                                                labels,
                                                label_list,
                                                device)

    train_metric.add_batch(
        predictions=preds,
        references=refs,
    )

    return


def run_eval_loop(dataloader, model, device,
                  metric, label_list, config):
    model.eval()
    for batch in tqdm(dataloader, desc='eval_loop'):
        # get the inputs;
        inputs = {k: v.to(device) for k, v in batch.items()}

        if config["ablation"]["run_ablation"]:
            inputs = utils.ablation(config, inputs)

        # forward + backward + optimize
        outputs = model(**inputs)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        preds, refs = utils.convert_preds_to_labels(predictions,
                                                    labels,
                                                    label_list,
                                                    device)

        metric.add_batch(predictions=preds, references=refs)


    return


def main(config):
    # get the  list of labels along with the label to id mapping and
    # reverse mapping
    label_list, id2label, label2id = utils.get_label_list(config)

    print("*" * 50)
    print('Prepared Label List. Preparing Training Data ')
    print("*" * 50)

    # preprocess the train and eval dataset
    train_data = utils.get_dataset(
        config["data"]["train_contract_dir"],
        id2label,
        label2id,
        data_split='train',
        num_contracts=config['data']['data_split']
    )

    print("*" * 50)
    print('Prepared Training Data. Preparing Eval Data ')
    print("*" * 50)

    eval_data = utils.get_dataset(
        config["data"]["eval_contract_dir"],
        id2label,
        label2id,
        data_split='eval',
        num_contracts=None
    )

    test_data = utils.get_dataset(
        config["data"]["test_contract_dir"],
        id2label,
        label2id,
        data_split='test',
        num_contracts=None
    )

    print("*" * 50)
    print(f'Using Large Model: {config["model"]["use_large_model"]}')
    print("*" * 50)

    # define the processor and model
    if config["model"]["use_large_model"]:
        processor = MarkupLMProcessor.from_pretrained(
            "microsoft/markuplm-large", only_label_first_subword=False
        )
        model = MarkupLMForTokenClassification.from_pretrained(
            "microsoft/markuplm-large", id2label=id2label, label2id=label2id
        )

    else:
        processor = MarkupLMProcessor.from_pretrained(
            "microsoft/markuplm-base", only_label_first_subword=False
        )
        model = MarkupLMForTokenClassification.from_pretrained(
            "microsoft/markuplm-base", id2label=id2label, label2id=label2id
        )

    processor.parse_html = False

    # convert the input dataset
    # to torch datasets. Create the dataloaders as well
    train_dataset = input_pipeline.MarkupLMDataset(
        data=train_data,
        processor=processor,
        max_length=config["model"]["max_length"]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["model"]["train_batch_size"],
        shuffle=True
    )

    eval_dataset = input_pipeline.MarkupLMDataset(
        data=eval_data,
        processor=processor,
        max_length=config["model"]["max_length"]
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["model"]["eval_batch_size"],
        shuffle=False
    )

    ## TEST Set
    test_dataset = input_pipeline.MarkupLMDataset(
        data=test_data,
        processor=processor,
        max_length=config["model"]["max_length"]
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    # get the class weights used to weigh the different terms in the loss fn
    class_value_counts, class_weights = utils.get_class_dist(
        config["data"]["train_contract_dir"], id2label, label2id
    )

    # define the optimizer and loss fct
    optimizer = AdamW(model.parameters(), lr=config["model"]["learning_rate"])

    loss_fct = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        ignore_index=config["model"]["ignore_index"]
    )

    # define the train and eval metric containers
    train_metric = evaluate.load("seqeval",
                                 scheme="BILOU",
                                 mode="strict",
                                 experiment_id="train",
                                 keep_in_memory=True)
    eval_metric = evaluate.load("seqeval",
                                scheme="BILOU",
                                mode="strict",
                                experiment_id='eval',
                                keep_in_memory=True)

    test_metric = evaluate.load("seqeval",
                                scheme="BILOU",
                                mode="strict",
                                experiment_id='test',
                                keep_in_memory=True)


    model.to(device)  # move to GPU if available
    num_epochs = config["model"]["num_epochs"]

    print("*" * 50)
    print(f'Running Training Loop for {num_epochs} epochs!')
    print("*" * 50)

    model.train()
    best_eval_score = -float("inf")
    num_epochs_lower_eval, best_epoch = 0, 0
    train_metrics_list, eval_metrics_list = [], []
    for epoch in range(num_epochs):
        model.train()
        for train_batch in tqdm(train_dataloader,
                                desc='train_loop'):
            run_train_loop(
                train_batch,
                model,
                optimizer,
                loss_fct,
                device,
                train_metric,
                label_list,
                config,
            )

        # run eval loop
        run_eval_loop(eval_dataloader, model, device,
                      eval_metric, label_list, config)

        # compute the metrics at the end of each epoch
        train_metrics = utils.compute_metrics(train_metric)
        eval_metrics = utils.compute_metrics(eval_metric)

        train_metrics['epoch'] = epoch
        train_metrics_list.append(train_metrics)

        eval_metrics['epoch'] = epoch
        eval_metrics_list.append(eval_metrics)

        # save the state dict for the best run
        if eval_metrics["overall_f1"] > best_eval_score:
            model_savepath = config["model"]["model_savepath"].rsplit(".", 1)[0]
            model_savepath = (
                f"{model_savepath}_epoch-{epoch}_f1-{eval_metrics['overall_f1']:0.3f}.pt"
            )

            print(f"Eval score improved. {best_eval_score} -> {eval_metrics['overall_f1']}")
            print(f"Saving ckpt at {model_savepath}")

            torch.save(model.state_dict(), model_savepath)

            # reset the patience counter for early stopping
            num_epochs_lower_eval = 0
            best_epoch = epoch
            best_eval_score = eval_metrics["overall_f1"]

        else:
            num_epochs_lower_eval += 1
            print(f"Eval f1 score did not improve. Patience={num_epochs_lower_eval}")

        print(
            f"Epoch {epoch} Train Metrics: {train_metrics}"
            + f"\n\nEval Metrics: {eval_metrics}"
        )

        if num_epochs_lower_eval >= config["model"]["early_stop_patience"]:
            print("*" * 50)
            print(f"Finished Training Early. Best Epoch {best_epoch} ")
            print("*" * 50)
            break

    print("*" * 50)
    print(f'Finished Training. Best Eval Score {best_eval_score}')
    print("*" * 50)


    print("Running Prediction Loop")
    run_eval_loop(test_dataloader, model, device,
                    test_metric, label_list, config)

    # compute the metrics at the end of each epoch
    test_metrics = utils.compute_metrics(test_metric)

    return model, train_metrics_list, eval_metrics_list, [test_metrics]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="Path to config file specifiying user params",
    )

    args = parser.parse_args()

    print("*" * 50)
    print('Processing...')
    print("*" * 50)

    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    collateral_dir = config['model']['collateral_dir']

    # if data split is provided then we're running training curve exps,
    # create the associate collateral dir
    data_split = config['data']['data_split']
    if data_split:
        collateral_dir = os.path.join(collateral_dir, f"num_contracts-{data_split}")

    if not os.path.exists(collateral_dir):
        os.makedirs(collateral_dir, exist_ok=True)

    model_dir = os.path.join(collateral_dir, 'ckpt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    config['model']['model_savepath'] = \
        os.path.join(model_dir,
                        config['model']['model_savepath'])


    model, train_metrics_list, eval_metrics_list, test_metrics = main(config)

    train_metrics_df = pd.DataFrame(train_metrics_list)
    eval_metrics_df = pd.DataFrame(eval_metrics_list)
    test_metrics_df = pd.DataFrame(test_metrics)

    from IPython.display import display

    display(eval_metrics_df)

    train_metrics_savepath = os.path.join(collateral_dir, "train_metrics.csv")
    train_metrics_df.to_csv(train_metrics_savepath, index=False)
    print(f"saved train metrics at {train_metrics_savepath}")

    eval_metrics_savepath = os.path.join(collateral_dir, "eval_metrics.csv")
    eval_metrics_df.to_csv(eval_metrics_savepath, index=False)
    print(f"saved eval metrics at {eval_metrics_savepath}")

    test_metrics_savepath = os.path.join(collateral_dir, "test_metrics.csv")
    test_metrics_df.to_csv(test_metrics_savepath, index=False)
    display(test_metrics_df)
    print(f"saved test metrics at {test_metrics_savepath}")