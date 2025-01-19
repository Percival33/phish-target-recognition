# Phish Target Recognition

- `poetry config virtualenvs.in-project true`
- `poetry env use $(pyenv which python)`
- `poetry install`
- `sudo xcode-select --switch /Applications/Xcode.app` from [here](https://github.com/PX4/PX4-SITL_gazebo-classic/issues/1021)


https://stackoverflow.com/questions/77250743/mac-xcode-g-cannot-compile-even-a-basic-c-program-issues-with-standard-libr

```sh
docker run -it --rm -v $(realpath ~/inz/src/models/visualphishnet):/tf/notebooks -p 8888:8888 --runtime=nvidia  tensorflow/tensorflow:1.14.0-gpu-py3-jupyter
```
```py
uv pip install tensorflow matplotlib wandb scikit-learn scikit-image jupyter pydantic_settings numpy tqdm
```
`python -m src.models.visualphishnet.evaluate`
`uv run -m src.models.visualphishnet.evaluate`