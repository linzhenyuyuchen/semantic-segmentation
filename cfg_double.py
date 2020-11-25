cfgs = {
    "n_classes" : 3,
    "gpus" : 1,
    "epochs" : 60,
    "batch_size_train" : 8,
    "net_name" : "clsresunet", # clsresunet
    "image_root" : "/data1/lzy/wsi/",
    "json_file_train" : "/data1/lzy/wsi/annotations3/cocojson_13_train_1.json",
    "json_file_val" : "/data1/lzy/wsi/annotations3/cocojson_13_val_1.json",
    "checkpoint_dir" : "/data1/lzy/wsi/checkpoint/clsresunet/",
    "checkpoint_space" : 1000,
    "class_names": {
            0: "bg" ,
            1: "low level" ,
            2: "high level" ,
        }
}
