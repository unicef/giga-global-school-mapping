import os
import argparse
import joblib
import pandas as pd
import logging
import torch

import sys
sys.path.insert(0, "../utils/")
import data_utils
import config_utils
import model_utils
import embed_utils
import eval_utils
import wandb

cwd = os.path.dirname(os.getcwd())

def main(iso, config):
    exp_name = f"{iso}_{config['config_name']}"
    wandb.run.name = exp_name
    results_dir = os.path.join(cwd, config["exp_dir"], c["project"], exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model = embed_utils.load_model(config)
    data = model_utils.load_data(config, attributes=["rurban", "iso"], verbose=True)
    columns = ["UID", "iso", "rurban", "dataset", "class"]

    out_dir = os.path.join(config["vectors_dir"], "embeddings")
    embeddings = embed_utils.get_image_embeddings(config, data, model, out_dir, in_dir=None, columns=columns)
    embeddings.columns = [str(x) for x in embeddings.columns]
    embeddings = embeddings[[x for x in embeddings.columns if x not in columns[1:]]]

    # Temporary fix
    embeddings = pd.merge(embeddings, data[columns], on="UID", how="inner")
    out_dir = data_utils._makedir(out_dir)
    filename = os.path.join(out_dir, f"{iso}_{model.name}_embeds.csv")
    embeddings.to_csv(filename, index=False)
    
    test = embeddings [embeddings.dataset == "test"]
    train = embeddings[(embeddings.dataset == "train") | (embeddings.dataset == "val")]
    logging.info(train.columns)

    logging.info(f"Test size: {test.shape}")
    logging.info(f"Train size: {train.shape}")
    
    target = "class"
    features = [str(x) for x in embeddings.columns if x not in columns]
    classes = list(embeddings[target].unique())
    logging.info(f"No. of features: {len(features)}")
    logging.info(f"Classes: {classes}")

    logging.info("Training model...")
    cv = model_utils.model_trainer(c, train, features, target)
    logging.info(f"Best estimator: {cv.best_estimator_}")
    logging.info(f"Best CV score: {cv.best_score_}")

    model = cv.best_estimator_
    model.fit(train[features], train[target].values)
    train_preds = model.predict(train[features])
    test_preds = model.predict(test[features])

    model_file = os.path.join(results_dir, f"{iso}_{config['config_name']}.pkl")
    joblib.dump(model, model_file)

    pred_col = "y_preds"
    train[pred_col] = train_preds
    test[pred_col] = test_preds
    pos_class = config["pos_class"]

    for phase in ["train", "test"]:
        result = train if phase == "train" else test
        eval_utils.save_results(
            result, 
            target=target, 
            pos_class=pos_class, 
            classes=classes, 
            pred=pred_col,
            results_dir=os.path.join(results_dir, phase),
            prefix=phase
        )

    phase = "test"    
    for rurban in ["urban", "rural"]:
        subresults_dir = os.path.join(results_dir, phase, rurban)
        subtest = test[test.rurban == rurban]
        results = eval_utils.save_results(
            subtest, 
            target=target, 
            pos_class=pos_class, 
            classes=classes, 
            pred=pred_col,
            results_dir=subresults_dir, 
            prefix=f"{phase}_{rurban}"
        )
            
      
if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--model_config", help="Config file")
    parser.add_argument("--iso", help="ISO code", default=[], nargs='+')
    args = parser.parse_args()

    # Load config
    config_file = os.path.join(cwd, args.model_config)
    c = config_utils.load_config(config_file)
    c["iso_codes"] = args.iso
    log_c = {
        key: val for key, val in c.items() 
        if ('url' not in key) 
        and ('dir' not in key)
        and ('file' not in key)
    }
    iso = args.iso[0]
    if "name" in c: iso = c["name"]
    log_c["iso_code"] = iso
    logging.info(log_c)
    
    wandb.init(
        project=c["project"],
        config=log_c,
        tags=[c["embed_model"], c["model"]]
    )

    main(iso, c)