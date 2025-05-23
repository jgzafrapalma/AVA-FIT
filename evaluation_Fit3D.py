import os
import roma
import torch
import joblib
import pickle
import argparse
import numpy as np

from utils.constants import SMPLX2SMPL_REGRESSOR, J_REGRESSOR_H36M


H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


with open(SMPLX2SMPL_REGRESSOR, 'rb') as f:
    smplx2smpl_regressor = torch.from_numpy(pickle.load(f)['matrix'].astype(np.float32)).to(device)

def evaluation_Fit3D(predictions_path, gt_smplx_path, output_path, 
                     participants = None, exercises = None, viewpoints = None):
    
    if participants:
        PARTICIPANT = participants
    else:
        PARTICIPANT = os.listdir(predictions_path)

    results = {}

    for participant in PARTICIPANT:

        if exercises:
            EXERCISES = exercises
        else:
            EXERCISES = [
                f for f in os.listdir(os.path.join(predictions_path, participant))
                if not f.startswith('.') and os.path.isdir(os.path.join(predictions_path, participant, f))
            ]

        for exercise in EXERCISES:

            if viewpoints:
                VIEWPOINTS = viewpoints
            else:
                VIEWPOINTS = os.listdir(os.path.join(predictions_path, participant, exercise))

            for viewpoint in VIEWPOINTS:

                print(f"Obteniendo resultados para {participant} - {exercise} - {viewpoint}")

                # Load SMPLX GT
                gt_smplx = np.load(os.path.join(gt_smplx_path, participant, exercise, viewpoint, 'gt.npz'))

                v3d = gt_smplx['v3d']
                pelvis = gt_smplx['transl_pelvis']

                # Load Predictions
                preds = np.load(os.path.join(predictions_path, participant, exercise, viewpoint, 'preds.npz'))

                v3d_hat = preds['v3d']
                pelvis_hat = preds['transl_pelvis']

                # TEMPORAL (BUSCAR OTRAS FORMAS DE SINCRONIZAR GT CON PREDICCIONES)
                indexes = list(map(int, preds['img_path']))
                pred_indexes = [i for i, x in enumerate(indexes) if x < v3d.shape[0]]
                gt_indexes = [x for x in indexes if x < v3d.shape[0]]

                paths_selected = preds['img_path'][pred_indexes]

                v3d = v3d[gt_indexes]
                pelvis = pelvis[gt_indexes]

                v3d_hat = v3d_hat[pred_indexes]
                pelvis_hat = pelvis_hat[pred_indexes]

                v3d_ctx = v3d - pelvis
                v3d_hat_ctx = v3d_hat - pelvis_hat

                # PVE and PA-PVE (SMPLX)

                if v3d_hat_ctx.shape[1] == 10475:

                    # Per-Vertex Error
                    pve = ((np.sqrt(((v3d_ctx - v3d_hat_ctx) ** 2).sum(-1))) * 1000).mean(axis=1)

                    pa_v3d_hat_ctx = compute_similarity_transform_batch(v3d_hat_ctx, v3d_ctx)
                    pa_pve = ((np.sqrt(((v3d_ctx - pa_v3d_hat_ctx) ** 2).sum(-1))) * 1000).mean(axis=1)

                    print(f"SMPLX: MPVPE: {pve.mean():.2f} - PA-MPVPE: {pa_pve.mean():.2f}")

                v3d_ctx = torch.from_numpy(v3d_ctx).to(device)
                v3d_hat_ctx = torch.from_numpy(v3d_hat_ctx).to(device)

                # Get SMPL vertices
                v3d_hat_ctx = smplx2smpl_regressor @ v3d_hat_ctx
                v3d_ctx = smplx2smpl_regressor @ v3d_ctx

                v3d_hat_ctx = v3d_hat_ctx.cpu().numpy()
                v3d_ctx = v3d_ctx.cpu().numpy()

                pve = ((np.sqrt(((v3d_ctx - v3d_hat_ctx) ** 2).sum(-1))) * 1000).mean(axis=1)

                pa_v3d_hat_ctx = compute_similarity_transform_batch(v3d_hat_ctx, v3d_ctx)
                pa_pve = ((np.sqrt(((v3d_ctx - pa_v3d_hat_ctx) ** 2).sum(-1))) * 1000).mean(axis=1)

                print(f"SMPL: MPVPE: {pve.mean():.2f} - PA-MPVPE: {pa_pve.mean():.2f}")

                J_regressor_h36m = torch.Tensor(np.load(J_REGRESSOR_H36M)).to(device)

                v3d_ctx = torch.from_numpy(v3d_ctx).to(device)
                v3d_hat_ctx = torch.from_numpy(v3d_hat_ctx).to(device)

                h36m = J_regressor_h36m @ v3d_ctx
                h36m_hat = J_regressor_h36m @ v3d_hat_ctx

                h36m = h36m.cpu().numpy()
                h36m_hat = h36m_hat.cpu().numpy()

                # center around h36m-pelvis
                h36m_ctx = h36m - h36m[:, [0], :]
                h36m_hat_ctx = h36m_hat - h36m_hat[:, [0], :]

                h36m_ctx = h36m_ctx[:, H36M_TO_J17, :]
                h36m_hat_ctx = h36m_hat_ctx[:, H36M_TO_J17, :]

                mpjpe = ((np.sqrt(((h36m_ctx - h36m_hat_ctx) ** 2).sum(-1))) * 1000).mean(axis=1)
                pa_h36m_hat_ctx = compute_similarity_transform_batch(h36m_hat_ctx, h36m_ctx)
                pa_mpjpe = ((np.sqrt(((h36m_ctx - pa_h36m_hat_ctx) ** 2).sum(-1))) * 1000).mean(axis=1)

                print(f"MPJPE: {mpjpe.mean():.2f} - PA-MPJPE: {pa_mpjpe.mean():.2f}")


                if participant not in results:
                    results[participant] = {}
                if exercise not in results[participant]:
                    results[participant][exercise] = {}
                if viewpoint not in results[participant][exercise]:
                    results[participant][exercise][viewpoint] = {}
                
                results[participant][exercise][viewpoint] = {"MPVPE": pve, "PA-MPVPE": pa_pve, "MPJPE": mpjpe, "PA-MPJPE": pa_mpjpe,
                                                             "paths": np.array(paths_selected)}
                
                #print(f"MPVPE: {pve.mean():.2f} - PA-MPVPE: {pa_pve.mean():.2f}")

    with open(os.path.join(output_path, 'results.pkl'), "wb") as f:
        joblib.dump(results, f)
            

def compute_similarity_transform(S1, S2):
    """
    Source of the code: https://gist.github.com/mkocabas/54ea2ff3b03260e3fedf8ad22536f427
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])
    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)
    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)
    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))
    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1
    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))
    # 7. Error:
    S1_hat = scale*R.dot(S1) + t
    if transposed:
        S1_hat = S1_hat.T
    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def get_args():
    parser = argparse.ArgumentParser(description="Get metrics from multi-hmr predictions")
    parser.add_argument("--preds_path", type=str, required=True, help="Paht to the predictions files")
    parser.add_argument("--gt_smplx_path", type=str, required=True, help="Path to the SMPLX vertices and pelvis GT")
    parser.add_argument("--participants", nargs='+', required=False, help="Participants to process")
    parser.add_argument("--viewpoints", nargs='+', required=False, help="Viewpoints to process")
    parser.add_argument("--exercises", nargs='+', required=False, help="Exercises to process")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the results")
    
    return parser.parse_args()

def main(args):

    PREDICTIONS_PATH = args.preds_path
    SMPLX_PATH = args.gt_smplx_path
    SAVE_PATH = args.save_path

    evaluation_Fit3D(PREDICTIONS_PATH, SMPLX_PATH, SAVE_PATH, args.participants, args.exercises, args.viewpoints)

if __name__ == "__main__":
    args = get_args()
    main(args)
