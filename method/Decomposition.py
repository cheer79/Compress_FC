import copy
import torch


def head_forward(head, x, rois, roi_indices, img_size):
    n, _, _, _ = x.shape
    if x.is_cuda:
        roi_indices = roi_indices.cuda()
        rois = rois.cuda()
    rois = torch.flatten(rois, 0, 1)
    roi_indices = torch.flatten(roi_indices, 0, 1)

    rois_feature_map = torch.zeros_like(rois)
    rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
    rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]

    indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
    pool = head.roi(x, indices_and_rois)
    fc7 = head.classifier(pool)
    fc7 = fc7.view(fc7.size(0), -1)

    return fc7

def Decomposition(layer, rank, model, dataloader, lambda1, lambda2, num_iter):
    """
    Creates model with bias corrected L1 Frobenius norm applied
    :param model: Initial model without decomposition
    :param rank: Number of singular vectors to keep (rank of the approximation)
    :param lambda1: Parameter of the L1 norm regular term
    :param lambda2: Parameter of the Frobenius norm regular term
    :return result_model: Model with compressed fc layers
    """

    with torch.no_grad():
        X = []

        for iteration, batch in enumerate(dataloader):
            images, boxes, labels = batch[0], batch[1], batch[2]
            images = images.cuda()
            img_size = images[0].shape[1:]

            base_feature = model.extractor.forward(images)
            _, _, rois, roi_indices, _ = model.rpn.forward(base_feature, img_size, 1.)
            output_tensor = head_forward(model.head, base_feature, rois, roi_indices, img_size)
            output_tensor = output_tensor.view(images[0].shape[0], -1, output_tensor.shape[1])
            output_tensor1 = output_tensor.mean(1)
            X.append(output_tensor1)
        X = torch.cat(X, 0)
        W = layer.weight
        W = torch.tensor(W)
        Z = W @ X.T
        z1 = torch.zeros(W.shape).cuda()
        Z_star = torch.cat((Z, z1), dim=1)
        lambda2 = torch.tensor(lambda2)
        x1 = torch.sqrt(lambda2) * torch.eye(X.shape[1])
        x1 = x1.cuda()
        X_star = (1 / torch.sqrt(1 + lambda2)) * torch.cat((X, x1), dim=0)  # (q+n)*n


        #初始化AB
        U, D, Vt = torch.linalg.svd(W)
        D1 = torch.diag(D)
        D2 = torch.sqrt(D1)
        A2 = U @ D2
        B2 = D2 @ Vt[:D1.shape[1],:]
        A = A2[:, :rank]
        B0 = B2[:rank, :]
        B = B0.T.cuda()


        B_star = torch.sqrt(1+lambda2) * B
        Gamma = lambda1/torch.sqrt(1+lambda2)

        for i in range(num_iter):
            B_start = B_star.T
            L = B_start.shape[1]
            B_start_end = torch.full_like(B_start, 0)
            for l in range(L):
                B0 = copy.deepcopy(B_start)
                BXr = (B0 @ X_star.T) - (B0[:, l].unsqueeze(1) @ X_star.T[l, :].unsqueeze(0))
                R = (A.T @ Z_star) - BXr
                X_start = X_star.T
                X_star_l = X_star[:, l]
                Sl = (2 / Gamma) * R @ X_star_l
                q = torch.linalg.norm(Sl.unsqueeze(1), 'fro')
                if q < 1:
                    B_start_end[:, l] = 0
                else:
                    X_star_l_normF = X_start[l, :] @ X_star[:, l]
                    R_X_star_l_norm1 = 2 * torch.linalg.norm(R @ X_star_l, ord=1, axis=None, keepdims=False)
                    b1 = 1 / X_star_l_normF
                    b2 = 1 - (Gamma / R_X_star_l_norm1)
                    b3 = torch.maximum(b2, torch.zeros(1).cuda())
                    B_startl = (b1 * b3) * R @ X_star[:, l]
                    B_start_end[:, l] = B_startl
            Bt_end = (1 / torch.sqrt(1 + lambda2)) * B_start_end
            B_star = B_start_end.T

            a1 = Z_star @ X_star @ B_star
            a2 = B_star.T @ X_star.T @ X_star @ B_star
            a3 = torch.linalg.inv(a2)
            A = a1 @ a3
    return A, Bt_end