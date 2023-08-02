def build_model(model_name, configs):
    if model_name == 'StreamingVectorNet':
        from .models import StreamingVectorNet
        return StreamingVectorNet(configs)