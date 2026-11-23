"""Prepare an auditable, opt-in libcom shadow-module experiment.

No source file is edited by this module. This is not an upstream or validated
model release, and is not imported by Renderello's production path.
"""
import hashlib

PINNED_LIBCOM_REVISION='32afffddf985e3c2832262d41960c8702b8c0fcb'
PINNED_SHADOW_SHA256='59e636774f0a1b7eb3c9912e453cf6ed01c92dbde1ff831ff8b874e1f03a870d'


def prepare_experimental_shadow_source(original):
    """Return new bytes plus a change ledger; reject unknown original bytes.

    Does not install, download, import dependencies, or load model weights.
    Removing the unused ControlLDM import is a SEPARATE preflight change and
    is deliberately not applied here, because it belongs to another file.
    """
    if not isinstance(original,bytes):raise ValueError('source must be original bytes')
    digest=hashlib.sha256(original).hexdigest()
    if digest!=PINNED_SHADOW_SHA256:raise ValueError('publisher source hash does not match pinned revision')
    text=original.decode('utf-8')
    replacements=[
        ('temp = pred_bbx[i,2]','temp = pred_bbx[i,2].clone()','clone scalar before overlapping tensor assignments'),
        ('cv2.fillPoly(bbx_mask[i].cpu().numpy().astype(np.uint8), [np.int32(box_points)], 1)',
         'box_array = bbx_mask[i].cpu().numpy().astype(np.uint8)\n                        cv2.fillPoly(box_array, [np.int32(box_points)], 1)\n                        bbx_mask[i] = torch.from_numpy(box_array).to(device=self.device, dtype=self.weight_dtype)',
         'retain filled uint8 box and explicitly copy back to model tensor'),
        ("mask_embeddings = batch['embeddings'].to(self.device, dtype=self.weight_dtype)",
         "base_mask_embeddings = batch['embeddings'].to(self.device, dtype=self.weight_dtype).clone()",'retain unconditioned base once per image'),
        ('# Mask embeddings adjustment\n                        mask_label',
         '# Mask embeddings adjustment: fresh base each denoising step\n                        mask_embeddings = base_mask_embeddings.clone()\n                        mask_label',
         'condition a fresh base once per denoising step'),
    ]
    ledger=[]
    for old,new,reason in replacements:
        if text.count(old)!=1:raise ValueError('expected one exact patch target')
        text=text.replace(old,new,1);ledger.append(reason)
    compile(text,'shadow-generation.py','exec')
    result=text.encode('utf-8')
    return result,dict(revision=PINNED_LIBCOM_REVISION,original_sha256=digest,experimental_sha256=hashlib.sha256(result).hexdigest(),changes=ledger,model_inference_tested=False,default_enabled=False)
