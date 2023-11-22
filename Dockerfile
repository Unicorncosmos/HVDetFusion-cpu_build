# Start with a base image that has the necessary dependencies installed.
FROM python:3.9-slim-buster

# Install mmcv, mmcv-full, mmdet, mmengine, mmsegmentation, onnx, onnxruntime, and pytorch
RUN pip install mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.28.1 mmengine==0.8.2 mmsegmentation==0.30.0 onnx==1.14.1 onnxruntime==1.16.0 torch==1.9.0+cpu -f https://download.pytorch.org/whl/cpu.html


# Copy the rest of the application code to the container.
COPY . .

#After Preparing the Dataset the execute generate the pickle file
CMD ["python","tools/create_data_hvdet.py"]

# Download checkpoints
RUN mkdir checkpoint && \
    wget -O ./checkpoint/backbone.pth https://drive.google.com/uc\?export\=download\&id\=1EKIFO0OhA_m5PFB3PoKBgA-sqUwdypNg

# Install gdown
RUN pip install gdown

# Download checkpoints
RUN mkdir -p ./tools/convter2onnx/onnx_output && \
    gdown --id 1Axj6HlAZ6hCEkWnqVesRDXjsE_LqSl_b -O ./tools/convter2onnx/onnx_output/onnx_stage1.onnx && \
    gdown --id 1U0TqBTz3v-zkgTfyVgCMmrg3Dmo7Fqcy -O ./tools/convter2onnx/onnx_output/onnx_stage1_1.onnx && \
    gdown --id 17WI0N9lyME1ZSfR4ftG_JcT5yYjkpEMs -O ./tools/convter2onnx/onnx_output/onnx_stage2.onnx && \
    gdown --id 1uv95hDg-KW7Cw0RG8w9NfWGQAdoi0YY0 -O ./tools/convter2onnx/onnx_output/onnx_stage3.onnx

CMD ["python","mmdet3d/models/necks/setup.py"]

# Set the command to run the application.
CMD [ "python", "HVDet_infer.py configs/hvdet/HVDetInfer_sim.py tools/convter2onnx/onnx_output --fuse-conv-bn --eval bbox" ]
