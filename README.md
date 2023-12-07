# RentMate
## Set-Up Environment 🌲 
### Install the necessary dependencies
1. Create an environmental file:
```sh
$ python -m venv venv
```

2.1. Activate the environment using Windows:
```sh
$ venv\Scripts\activate
```

2.2. Activate the environment using Linux or MacOS:
```sh
$ source venv/bin/activate
```

3. Install PyTorch locally getting the commands from ([PyTorch - Get Started](https://pytorch.org/get-started/locally/)):
```sh
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. Install the requirements:
```sh
$ pip install -r requirements.txt
```