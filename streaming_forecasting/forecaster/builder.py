def build_model(model_name, model_config):
    if model_name == 'VectorNet':
        from .models import VectorNet
        model = VectorNet
    return model


def build_dataset(model_name, dataset_config):
    if model_name == 'VectorNet':
        from .dataset import ArgoDataset, collate_fn
        dataset = ArgoDataset
        collate_fn = collate_fn
    return dataset, collate_fn


def build_loss():
    from .models.loss import JointLoss
    loss = JointLoss
    return loss


def builder(model_name, dataset_config, model_config):
    model = build_model(model_name, model_config)
    dataset, collate_fn = build_dataset(model_name, dataset_config)
    loss = build_loss()
    return dataset, collate_fn, model, loss