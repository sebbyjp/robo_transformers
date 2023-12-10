# Instructions:

- Run `pip install -r requirements.xt`
- Install protobuf compiler (see [instructions](https://grpc.io/docs/protoc-installation/))
- Compile t2r.proto:  `protoc -I=tensor2robot/proto --python_out=tensor2robot/proto tensor2robot/proto/t2r.proto`
- Install git-lfs
- Run `git lfs install`
- Run `git lfs clone https://www.github.com/google-research/robotics_transformer.git `
<!-- - Run `export PYTHONPATH=.` or `pip install -e tensor2robot` -->