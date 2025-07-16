import gigaam

onnx_dir = "onnx3"
model_type = "emo"
download_root_path = "path_to_gigaam_emo"
model = gigaam.load_model(
    model_type,
    fp16_encoder=False,  # only fp32 tensors
    use_flash=False,  # disable flash attention
    download_root=download_root_path
    )
model.to_onnx(dir_path=onnx_dir)