import torch
import torch.nn as nn


class MultiDomainMaskDecoder(nn.Module):
    def __init__(self, decoder_factory, num_domains=3):
        super().__init__()
        self.num_domains = num_domains
        self.decoders = nn.ModuleList([
            decoder_factory() for _ in range(num_domains)
        ])

    def _select_batch(self, x, idxs, batch_size):
        if x is None:
            return None
        if not torch.is_tensor(x):
            return x
        if x.dim() > 0 and x.size(0) == batch_size:
            return x[idxs]
        return x

    def _domain_preprocess_high_res_features(self, high_res_features, idxs, domain_id, batch_size):
        """
        Apply domain-specific conv_s0 / conv_s1 if the decoder has them.
        high_res_features is expected to be a list/tuple like:
            [feat_s0, feat_s1]
        where each feat is [B, C, H, W]
        """
        if high_res_features is None:
            return None

        decoder = self.decoders[domain_id]

        if not isinstance(high_res_features, (list, tuple)):
            # 如果不是 list/tuple，就直接按 batch 切
            return self._select_batch(high_res_features, idxs, batch_size)

        feats = [self._select_batch(feat, idxs, batch_size) for feat in high_res_features]

        # 如果单个 decoder 里有 conv_s0 / conv_s1，则对对应域样本做各自处理
        if len(feats) > 0 and hasattr(decoder, "conv_s0") and feats[0] is not None:
            feats[0] = decoder.conv_s0(feats[0])

        if len(feats) > 1 and hasattr(decoder, "conv_s1") and feats[1] is not None:
            feats[1] = decoder.conv_s1(feats[1])

        return feats

    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=None,
        clss=None,
    ):
        assert clss is not None, "clss must be provided"
        B = image_embeddings.shape[0]

        masks_out = [None] * B
        iou_out = [None] * B
        token_out = [None] * B
        score_out = [None] * B

        for d in range(self.num_domains):
            mask = (clss == d)
            if not mask.any():
                continue

            idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)

            img_emb_d = self._select_batch(image_embeddings, idxs, B)
            img_pe_d = self._select_batch(image_pe, idxs, B)
            sparse_d = self._select_batch(sparse_prompt_embeddings, idxs, B)
            dense_d = self._select_batch(dense_prompt_embeddings, idxs, B)

            # 关键：高分辨率特征按域分别走对应 expert 的 conv_s0 / conv_s1
            high_res_d = self._domain_preprocess_high_res_features(
                high_res_features=high_res_features,
                idxs=idxs,
                domain_id=d,
                batch_size=B,
            )

            masks, iou_pred, mask_tokens_out, object_score_logits = self.decoders[d](
                image_embeddings=img_emb_d,
                image_pe=img_pe_d,
                sparse_prompt_embeddings=sparse_d,
                dense_prompt_embeddings=dense_d,
                multimask_output=multimask_output,
                repeat_image=repeat_image,
                high_res_features=high_res_d,
            )

            for local_i, global_i in enumerate(idxs.tolist()):
                masks_out[global_i] = masks[local_i]
                iou_out[global_i] = iou_pred[local_i]
                token_out[global_i] = mask_tokens_out[local_i]
                score_out[global_i] = object_score_logits[local_i]

        masks_out = torch.stack(masks_out, dim=0)
        iou_out = torch.stack(iou_out, dim=0)
        token_out = torch.stack(token_out, dim=0)
        score_out = torch.stack(score_out, dim=0)

        return masks_out, iou_out, token_out, score_out