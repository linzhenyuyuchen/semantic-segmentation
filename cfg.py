cfgs = {
    "n_classes" : 2,
    "gpus" : 1,
    "epochs" : 20,
    "batch_size_train" : 8,
    "net_name" : "U_Net", # deeplab,unet,U_Net,R2U_Net,AttU_Net,R2AttU_Net
    "image_root" : "/data2/lzy/pathology/train2017/",
    "json_file_train" : "/data2/lzy/pathology/annotations3/cocojson_13_train_1.json",
    "json_file_val" : "/data2/lzy/pathology/annotations3/cocojson_13_val_1.json",
    "checkpoint_dir" : "/data2/lzy/pathology/checkpoint/",
    "checkpoint_space" : 2000,
    "class_names": {
            0: "bg" ,
            1: "liver" ,
            2: "left_kidney" ,
            3: "right_kidney" ,
            4: "spleen"
        }
}
