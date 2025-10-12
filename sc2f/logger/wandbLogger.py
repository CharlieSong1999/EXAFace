import wandb
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO
from typing import List, Dict, Tuple
from collections import Counter
from models.expansion.rope_attn import MultiheadAttention_with_pos_embed

class WandbAttentionLogger:
    def __init__(self, run, dataset, transform, selected_indices: List[int], cfg: Dict = None, skip_attention_map=False):
        """
        Args:
            run: wandb run object
            dataset: dataset object with __getitem__ method
            transform: transformation to apply to images
            selected_indices: list of indices to log attention maps for
            cfg: configuration dictionary, optional
        """
        self.dataset = dataset
        self.selected_indices = selected_indices
        self.run = run
        self.transform = transform
        self.cfg = cfg
        self.skip_attention_map = skip_attention_map

    def _render_attention_map(self, img: Image.Image, attn: torch.Tensor, token_map: torch.Tensor,
                              token_idx: int, H_feat: int, W_feat: int, EAD=True) -> Image.Image:
        attn = attn.mean(0).sigmoid().numpy()
        assert attn.size == H_feat * W_feat, f"Attention map size mismatch: {attn.size} vs {H_feat * W_feat}"

        attn = attn.reshape(H_feat, W_feat)
        attn_resized = cv2.resize(attn, img.size[::-1], interpolation=cv2.INTER_CUBIC)
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.ptp() + 1e-6)
        heatmap_color = cm.get_cmap("jet")(attn_resized)[..., :3]
        heatmap_img = (heatmap_color * 255).astype(np.uint8)

        base_np = np.array(img)
        # Ensure both images are the same size
        if heatmap_img.shape != base_np.shape:
            heatmap_img = cv2.resize(heatmap_img, (base_np.shape[1], base_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(base_np, 0.5, heatmap_img, 0.5, 0)
        
        H, W = base_np.shape[:2]
        if EAD:
            ori_overlay = overlay.copy()
            overlay = np.ones((H * 3, W * 3, 3), dtype=np.uint8) * 30
            overlay[H:2*H, W:2*W, :] = ori_overlay
            del ori_overlay
        

        token_overlay = self._draw_token_overlay(token_map, token_idx, overlay.shape[:2])
        final = cv2.addWeighted(overlay, 1.0, token_overlay, 0.5, 0)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(final)
        ax.axis("off")
        fig.colorbar(cm.ScalarMappable(cmap="jet"), ax=ax, shrink=0.7, orientation='vertical')

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def _draw_token_overlay(self, token_map: torch.Tensor, token_idx: int, canvas_shape: Tuple[int, int]) -> np.ndarray:
        token_map_np = token_map.cpu().numpy()
        H_feat, W_feat = token_map_np.shape
        mask = (token_map_np == token_idx)
        overlay = np.zeros((H_feat, W_feat, 3), dtype=np.uint8)
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            overlay[y, x] = (255, 255, 0)
        overlay = cv2.resize(overlay, canvas_shape[::-1], interpolation=cv2.INTER_NEAREST)

        step_y, step_x = overlay.shape[0] // H_feat, overlay.shape[1] // W_feat
        for i in range(1, H_feat+1):
            cv2.line(overlay, (0, i * step_y), (overlay.shape[1], i * step_y), (100, 100, 100), 1)
        for j in range(1, W_feat+1):
            cv2.line(overlay, (j * step_x, 0), (j * step_x, overlay.shape[0]), (100, 100, 100), 1)
        return overlay
    
    def _register_attention_hooks(self, model):
        self.attn_store = {}
        hook_handlers = []
        
        def _make_hook(idx: int):
            def hook(_module, _inp, output):  # output: tuple(attn_out, attn_weights)
                # Many libraries return (context, attn_weights); adapt as needed.
                if isinstance(output, tuple):
                    weights = output[1]
                else:  # huggingface format attn returns weights directly
                    weights = output
                # Store CPU tensor for minimal memory footprint.
                # print(f"[DEBUG] Storing attention weights for layer {idx}: shape={weights.shape}")
                if idx not in self.attn_store:
                    self.attn_store[idx] = [weights.detach().cpu()]
                else:
                    self.attn_store[idx].append(weights.detach().cpu())
                    self.attn_store[idx] = [torch.cat(self.attn_store[idx], dim=1) if len(self.attn_store[idx]) > 1 else self.attn_store[idx][0]]
            return hook

        layer_idx = 0
        num_cross_attn_layers = 0
        for name, module in model.named_modules():
            if (isinstance(module, torch.nn.MultiheadAttention) or isinstance(module, MultiheadAttention_with_pos_embed)) and 'cross_attn' in name:
                num_cross_attn_layers += 1
                hook_handlers.append(module.register_forward_hook(_make_hook(layer_idx)))
                # module.register_forward_hook(_debug_hook(name))  # Optional debug hook
                print(f"Registered attention hook on {name} (module {module}) with layer index {layer_idx}")
                
                layer_idx += 1
                
                if 'expansion_cfg' in self.cfg.keys() and 'attn' in self.cfg['expansion_cfg'].keys() and \
                   'decoder_per_granularity' in self.cfg['expansion_cfg']['attn'].keys() and \
                   self.cfg['expansion_cfg']['attn']['decoder_per_granularity']:
                    if num_cross_attn_layers % self.cfg['expansion_cfg']['attn']['num_dec_layers'] == 0:
                        layer_idx = 0
                
                    
                
        return hook_handlers

    def log(self, model, step: int):
        model.eval()
        step = self.run.step
        
        ids = []
        
        # Register hooks to capture attention weights
        hook_handlers = self._register_attention_hooks(model)
        model.expansion.attn_fest_query.need_attn_weights = True  # Enable attention weights logging
        
        for idx in self.selected_indices:
            
            
            img, id_ = self.dataset.pull_image(idx)
            img = img[:, :, ::-1]  # Convert BGR to RGB if needed
            pil_img = T.ToPILImage()(img)
            h, w, _ = img.shape
            orig_size = np.array([[w, h, w, h]])
            
            # preprocess
            x = self.transform(img)[0]
            x = x.unsqueeze(0).to(model.device)
            
            id_ = int(id_)
            ids.append(id_)
            
            with torch.no_grad():
                outputs = model(x)
                if isinstance(outputs, dict):
                    bboxes = outputs['bboxes']
                    scores = outputs['scores']
                    cls_inds = outputs['labels']
                    token_idx = outputs.get('token_idx', None)
                    if 'token_idx_manager' in outputs.keys():
                        token_map = outputs['token_idx_manager'].token_id_map
                        token_map = token_map[0, ...].detach().cpu() if token_map is not None else None
                    else:
                        token_map = model.expanded_token_index if hasattr(model, 'expanded_token_index') else None
                        token_map = token_map[0, :, :, 0].detach().cpu() if token_map is not None else None
                else:
                    bboxes, scores, cls_inds = outputs
                    token_idx = None
                    token_map = None
                
            # rescale
            if model.EAD:
                orig_size = orig_size * 3
                ead_pil_img = torch.zeros((3, h * 3, w * 3), dtype=torch.float32)
                ead_pil_img[:, h:2*h, w:2*w] = T.ToTensor()(pil_img)
                ead_pil_img = T.ToPILImage()(ead_pil_img)
            # bboxes *= orig_size # Wandb Image does not need to be rescaled
            
            dets = []
            tokens_with_boxes = []
            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                label = self.dataset.class_ids[int(cls_inds[i])]
                
                bbox = [x1, y1, x2, y2]
                score = float(scores[i])
                dets.append({"bbox": bbox, "label": label, "score": score, "token_idx": int(token_idx[i]) if token_idx is not None else None})
                tokens_with_boxes.append(int(token_idx[i]) if token_idx is not None and token_idx[i] >= 0 else None)
                
            if len(tokens_with_boxes) >= 50:
                print(f"Warning: More than 50 tokens with boxes detected for image {idx}. Only the first 50 will be logged.")
                count = Counter(tokens_with_boxes)
                tokens_with_boxes = [item for item, _ in count.most_common(50)]
                # tokens_with_boxes = [token if token in top_50_tokens else None for token in tokens_with_boxes]

            # log bounding boxes
            box_list = []
            label_dict = {0: 'face', 1: 'person'}
            for i, d in enumerate(dets):
                x1, y1, x2, y2 = d["bbox"]
                box_list.append({
                    "position": {"minX": x1, "minY": y1, "maxX": x2, "maxY": y2},
                    "class_id": d["label"],
                    "scores": {"score": d["score"]}
                })
                # label_dict[d["label"]] = f"class_{d['label']}"

            self.run.log({
                f"detection/img_{idx}": wandb.Image(
                    ead_pil_img if model.EAD else pil_img, 
                    boxes={
                        "predictions": {
                            "box_data": box_list,
                            "class_labels": label_dict
                        }
                    }
                )
            }, step=step)

            # log attention map for out-of-crop tokens
            if not self.skip_attention_map:
                h_x, w_x = x.shape[-2:]
                H_feat, W_feat = h_x // 16, w_x // 16
                for layer, attn in self.attn_store.items():
                    attn = attn[0]  # Get the first element since we store a list of tensors
                    for token_id in tokens_with_boxes:
                        if token_id is None:
                            continue
                        attn_q = attn[0][:, token_id, :] if attn[0].ndim == 3 else attn[0][token_id, :].unsqueeze(0)
                        vis_img = self._render_attention_map(pil_img, attn_q, token_map, token_id, H_feat, W_feat)
                        self.run.log({
                            f"attention/img_{idx}/layer_{layer}/token_{token_id}": wandb.Image(vis_img)
                        }, step=step)
                    
        # Register hooks to capture attention weights
        for handler in hook_handlers:
            handler.remove()
        model.expansion.attn_fest_query.need_attn_weights = False
