import os
import pathlib
import torch
import joblib
import pickle
import argparse
import numpy as np

from utils.constants import SMPLX2SMPL_REGRESSOR, J_REGRESSOR_H36M, SMPLX_SEGMENTATION_JSON

HANDS_REGIONS = ['rightHand', 'rightHandIndex1', 'leftHand', 'leftHandIndex1']
FEETS_REGIONS = ['leftToeBase', 'leftFoot', 'rightFoot', 'rightToeBase']

H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


with open(SMPLX2SMPL_REGRESSOR, 'rb') as f:
    smplx2smpl_regressor = torch.from_numpy(pickle.load(f)['matrix'].astype(np.float32)).to(device)

def get_args():
    parser = argparse.ArgumentParser(description="Get metrics from multi-hmr predictions")
    parser.add_argument("--preds_path", type=str, required=True, help="Path to the predictions files")
    parser.add_argument("--gt_smplx_path", type=str, required=True, help="Path to the SMPLX vertices and pelvis GT")
    parser.add_argument("--participants", nargs='+', required=False, help="Participants to process")
    parser.add_argument("--viewpoints", nargs='+', required=False, help="Viewpoints to process")
    parser.add_argument("--exercises", nargs='+', required=False, help="Exercises to process")
    parser.add_argument("--repetitions", nargs='+', required=False, help="Repetitions to process")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the results")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    
    return parser.parse_args()

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

def evaluation_AVAFIT(predictions_path, gt_smplx_path, output_path, 
                     participants = None, exercises = None, viewpoints = None,
                     repetitions = None, overwrite = False):

    # Comprobar si existe el fichero de resultados
    if os.path.exists(os.path.join(output_path, 'results.pkl')) and not overwrite:
        print("El fichero de resultados ya existe. Saltando evaluación.")
        return
    
    if participants:
        PARTICIPANT = participants
    else:
        PARTICIPANT = os.listdir(predictions_path)

    results = {}

    for participant in sorted(PARTICIPANT):

        if exercises:
            EXERCISES = exercises
        else:
            EXERCISES = [
                f for f in os.listdir(os.path.join(predictions_path, participant))
                if not f.startswith('.') and os.path.isdir(os.path.join(predictions_path, participant, f))
            ]

        for exercise in sorted(EXERCISES):

            if viewpoints:
                VIEWPOINTS = viewpoints
            else:
                VIEWPOINTS = os.listdir(os.path.join(predictions_path, participant, exercise))

            for viewpoint in sorted(VIEWPOINTS):

                if repetitions:
                    REPETITIONS = repetitions
                else:
                    REPETITIONS = os.listdir(os.path.join(predictions_path, participant, exercise, viewpoint))

                for repetition in sorted(REPETITIONS):

                    if participant not in results:
                        results[participant] = {}
                    if exercise not in results[participant]:
                        results[participant][exercise] = {}
                    if viewpoint not in results[participant][exercise]:
                        results[participant][exercise][viewpoint] = {}
                    if repetition not in results[participant][exercise][viewpoint]:
                        results[participant][exercise][viewpoint][repetition] = {}

                    print(f"Obteniendo resultados para {participant} - {exercise} - {viewpoint} - {repetition}")

                    # Load SMPLX GT
                    gt_smplx = np.load(os.path.join(gt_smplx_path, participant, exercise, viewpoint, repetition, 'gt.npz'))

                    v3d = gt_smplx['v3d']
                    pelvis = gt_smplx['transl_pelvis']

                    # Load Predictions
                    preds = np.load(os.path.join(predictions_path, participant, exercise, viewpoint, repetition, 'preds.npz'))

                    v3d_hat = preds['v3d']
                    pelvis_hat = preds['transl_pelvis']

                    indexes = np.array(list(map(int, preds['img_path'])))
                    indexes = indexes - 1

                    paths_selected = preds['img_path']
                    
                    v3d = v3d[indexes]
                    pelvis = pelvis[indexes]

                    v3d_ctx = v3d - pelvis
                    v3d_hat_ctx = v3d_hat - pelvis_hat

                    smplx_pred = False

                    # FULL-BODY PVE and PA-PVE (SMPLX)
                    if v3d_hat_ctx.shape[1] == 10475:

                        smplx_pred = True

                        # Per-Vertex Error
                        pve_full = ((np.sqrt(((v3d_ctx - v3d_hat_ctx) ** 2).sum(-1))) * 1000).mean(axis=1)

                        pa_v3d_hat_ctx = compute_similarity_transform_batch(v3d_hat_ctx, v3d_ctx)
                        pa_pve_full = ((np.sqrt(((v3d_ctx - pa_v3d_hat_ctx) ** 2).sum(-1))) * 1000).mean(axis=1)

                        if args.verbose:
                            print(f"SMPLX: MPVPE: {pve_full.mean():.2f} - PA-MPVPE: {pa_pve_full.mean():.2f}")

                        # REGIONS ERRORS (SMPLX)
                        with open(SMPLX_SEGMENTATION_JSON, 'r') as f:
                            import json
                            segmentation_dict = json.load(f)
                        
                        region_errors = {}
                        hands_vertices = []
                        feets_vertices = []
                        # Calcular el error para cada región
                        for region, vertices in segmentation_dict.items():

                            if region in HANDS_REGIONS:
                                hands_vertices.extend(vertices)
                            elif region in FEETS_REGIONS:
                                feets_vertices.extend(vertices)
                            
                            # GT de la región
                            region_v3d = v3d_ctx[:, vertices, :]  
                            
                            # Predicción
                            region_v3d_hat = v3d_hat_ctx[:, vertices, :]
                            
                            # Predicción CON alineación GLOBAL (para PA-MPVPE)
                            # Usamos los vértices de la malla que YA fue alineada globalmente
                            region_pa_v3d_hat_global = pa_v3d_hat_ctx[:, vertices, :]

                            # 1. MPVPE (Error sin alineación)
                            region_pve = ((np.sqrt(((region_v3d - region_v3d_hat) ** 2).sum(-1))) * 1000).mean(axis=1)
                            
                            # 2. PA-MPVPE (Usando alineación global)
                            region_pa_pve = ((np.sqrt(((region_v3d - region_pa_v3d_hat_global) ** 2).sum(-1))) * 1000).mean(axis=1)

                            # Guardamos los errores
                            region_errors[region] = {"MPVPE_SMPLX": region_pve, "PA-MPVPE_SMPLX": region_pa_pve}

                            if args.verbose:
                                print(f"Error para la región {region}: MPVPE: {region_pve.mean():.2f} - PA-MPVPE: {region_pa_pve.mean():.2f}")


                        # Hands and Feets MPVPE and PA-MPVPE
                        hands_v3d = v3d_ctx[:, hands_vertices, :]
                        hands_v3d_hat = v3d_hat_ctx[:, hands_vertices, :]
                        hands_pa_v3d_hat_global = pa_v3d_hat_ctx[:, hands_vertices, :]

                        hands_pve = ((np.sqrt(((hands_v3d - hands_v3d_hat) ** 2).sum(-1))) * 1000).mean(axis=1)
                        hands_pa_pve = ((np.sqrt(((hands_v3d - hands_pa_v3d_hat_global) ** 2).sum(-1))) * 1000).mean(axis=1)

                        if args.verbose:
                            print(f"Hands: MPVPE: {hands_pve.mean():.2f} - PA-MPVPE: {hands_pa_pve.mean():.2f}")

                        feets_v3d = v3d_ctx[:, feets_vertices, :]
                        feets_v3d_hat = v3d_hat_ctx[:, feets_vertices, :]
                        feets_pa_v3d_hat_global = pa_v3d_hat_ctx[:, feets_vertices, :]

                        feets_pve = ((np.sqrt(((feets_v3d - feets_v3d_hat) ** 2).sum(-1))) * 1000).mean(axis=1)
                        feets_pa_pve = ((np.sqrt(((feets_v3d - feets_pa_v3d_hat_global) ** 2).sum(-1))) * 1000).mean(axis=1)

                        if args.verbose:
                            print(f"Feets: MPVPE: {feets_pve.mean():.2f} - PA-MPVPE: {feets_pa_pve.mean():.2f}")

                        # Get SMPL vertices from predictions
                        v3d_hat_ctx = torch.from_numpy(v3d_hat_ctx).to(device)
                        v3d_hat_ctx = smplx2smpl_regressor @ v3d_hat_ctx
                        v3d_hat_ctx = v3d_hat_ctx.cpu().numpy()
                        
                    # GT to SMPL
                    v3d_ctx = torch.from_numpy(v3d_ctx).to(device)
                    v3d_ctx = smplx2smpl_regressor @ v3d_ctx
                    v3d_ctx = v3d_ctx.cpu().numpy()

                    # ONLY-BODY (SMPL) 6890 vertices

                    pve = ((np.sqrt(((v3d_ctx - v3d_hat_ctx) ** 2).sum(-1))) * 1000).mean(axis=1)

                    pa_v3d_hat_ctx = compute_similarity_transform_batch(v3d_hat_ctx, v3d_ctx)
                    pa_pve = ((np.sqrt(((v3d_ctx - pa_v3d_hat_ctx) ** 2).sum(-1))) * 1000).mean(axis=1)

                    if args.verbose:
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

                    if args.verbose:
                        print(f"MPJPE: {mpjpe.mean():.2f} - PA-MPJPE: {pa_mpjpe.mean():.2f}")

                    if not smplx_pred:
                        results[participant][exercise][viewpoint][repetition] = {"MPJPE": mpjpe, "PA-MPJPE": pa_mpjpe, "MPVPE": pve, "PA-MPVPE": pa_pve,
                                                                "paths": np.array(paths_selected)}
                    else:
                        results[participant][exercise][viewpoint][repetition] = {"MPJPE": mpjpe, "PA-MPJPE": pa_mpjpe, "MPVPE": pve, "PA-MPVPE": pa_pve,
                                                                "MPVPE_SMPLX": pve_full, "PA-MPVPE_SMPLX": pa_pve_full, "MPVPE_HANDS": hands_pve, "PA-MPVPE_HANDS": hands_pa_pve, "MPVPE_FEETS": feets_pve, "PA-MPVPE_FEETS": feets_pa_pve,
                                                                "paths": np.array(paths_selected)}
                        results[participant][exercise][viewpoint]["region_errors"] = region_errors
                    
                    
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(output_path, 'results.pkl'), "wb") as f:
        joblib.dump(results, f)
                
def main(args):

    PREDICTIONS_PATH = args.preds_path
    SMPLX_PATH = args.gt_smplx_path
    SAVE_PATH = args.save_path

    evaluation_AVAFIT(PREDICTIONS_PATH, SMPLX_PATH, SAVE_PATH, args.participants, args.exercises, args.viewpoints, args.repetitions, args.overwrite)

if __name__ == "__main__":
    args = get_args()
    main(args)
