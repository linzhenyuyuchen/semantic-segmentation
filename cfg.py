cfgs = {
    "n_classes" : 4,
    "gpus" : 1,
    "epochs" : 20,
    "batch_size_train" : 8,
    "net_name" : "U_Net", # deeplab,unet,U_Net,R2U_Net,AttU_Net,R2AttU_Net
    "image_root" : "/data1/lzy/wsi/",
    "json_file_train" : "/data1/lzy/wsi/annotations3/cocojson_13_train_1.json",
    "json_file_val" : "/data1/lzy/wsi/annotations3/cocojson_13_val_1.json",
    "checkpoint_dir" : "/data1/lzy/wsi/checkpoint/",
    "checkpoint_space" : 2000,
    "class_names": {
            0: "bg" ,
            1: "normal" ,
            2: "low level" ,
            3: "high level" ,
        }
}
