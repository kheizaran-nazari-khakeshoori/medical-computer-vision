"""Saliency map generation for explainability."""



def saliency_map(model, image_tensor, class_idx=None):
    model.eval()
    image_tensor.requires_grad = True
    out = model(image_tensor)
    idx = class_idx or out.argmax(1).item()
    model.zero_grad()
    out[0, idx].backward()
    saliency = image_tensor.grad.abs().max(dim=1)[0].cpu().numpy()[0]
    saliency = (saliency - saliency.min()) / (saliency.max() + 1e-8)
    return saliency
