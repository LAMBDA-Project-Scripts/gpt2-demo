LLM Demo
========

A demo of word prediction probabilities for GPT-2.


Installation
------------
It is recommended to first create a virtual environment.

In a regular setup you should be able to create a proper environment with the
commands:

```
python -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

In my specific setup, however, you need to install a specific version of 
PyTorch with the command:

```
pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 flask --extra-index-url https://download.pytorch.org/whl/cu113
```

Running
-------
You can run the server with the following commands:

```
source venv/bin/activate
flask --app server run
```

This will start a webserver that you can access with your web browser of choice
and typing `http://localhost:5000`.
