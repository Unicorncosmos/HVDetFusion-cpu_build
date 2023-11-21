# Start with a base image that has the necessary dependencies installed.
FROM python:3.9-slim-buster

# Copy the requirements.txt file to the container.
COPY requirements.txt .

# Install the Python packages listed in requirements.txt using pip.
RUN pip install --no-cache-dir -r requirements.txt

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



# Set the command to run the application.
CMD [ "python", "HVDet_infer.py configs/hvdet/HVDetInfer_sim.py tools/convter2onnx/onnx_output --fuse-conv-bn --eval bbox" ]
