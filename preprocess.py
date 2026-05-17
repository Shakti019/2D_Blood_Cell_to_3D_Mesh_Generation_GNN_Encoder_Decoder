import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
import io
import base64
import torch
from torch_geometric.data import Data


def _encode_cv2(img_bgr):
    """Encode a BGR cv2 image to a base64 data URI."""
    _, buf = cv2.imencode('.png', img_bgr)
    return 'data:image/png;base64,' + base64.b64encode(buf).decode()


def _encode_fig(fig, facecolor='white'):
    """Encode a matplotlib figure to a base64 data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=facecolor, dpi=120)
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def process_image(image_bytes, line_step=15):
    """
    Given raw image bytes, extract nucleus mesh and return graph data + solid mesh + stage images.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    stages = []

    # --- Stage 1: Original Image ---
    stages.append({
        "title": "Step 1 — Original 2D Blood Cell Image",
        "desc": "Raw microscopy image. The nucleus stain contains the chromatin structure used for classification.",
        "img": _encode_cv2(img)
    })

    # 1. Isolate nucleus natively (DataPreprocessing pipeline)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]

    blurred_s = cv2.GaussianBlur(s_channel, (9, 9), 0)
    _, thresh = cv2.threshold(blurred_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Could not detect continuous boundaries.")

    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(thresh_cleaned)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    y_true, x_true = np.where(mask == 255)
    if len(x_true) == 0:
        raise ValueError("No nucleus found.")

    _, _, w, h = cv2.boundingRect(largest_contour)
    r_size = float(max(w, h) / 2)

    # 2. Extract pixel data arrays
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    absolute_rgb_colors = img_rgb[y_true, x_true]

    cx_true = x_true - np.median(x_true)
    cy_true = y_true - np.median(y_true)

    # 3. Subsample for 3D projection
    cx_l = cx_true[::line_step]
    cy_l = cy_true[::line_step]
    colors_float = absolute_rgb_colors[::line_step] / 255.0  # shape (N, 3)
    N = len(cx_l)

    # 4. 6-face CUBIC SPATIAL design — exact match to notebook faces_config
    #    Each face is a FLAT PANEL at distance r_size from centre on one axis.
    #    Front/Back  → constant Z  at ±r_size
    #    Right/Left  → constant X  at ±r_size
    #    Top/Bottom  → constant Y  at ±r_size
    faces_config = [
        ('Front',  cx_l.copy(),                            -cy_l.copy(),                           np.full(N, r_size,  dtype=float)),
        ('Back',   -cx_l.copy(),                            cy_l.copy(),                           np.full(N, -r_size, dtype=float)),
        ('Right',  np.full(N,  r_size, dtype=float),       -cy_l.copy(),                          -cx_l.copy()),
        ('Left',   np.full(N, -r_size, dtype=float),       -cy_l.copy(),                           cx_l.copy()),
        ('Top',    cx_l.copy(),   np.full(N,  r_size, dtype=float),                               -cy_l.copy()),
        ('Bottom', -cx_l.copy(),  np.full(N, -r_size, dtype=float),                                cy_l.copy()),
    ]

    # 5. Delaunay triangulation on the 2-D nucleus footprint
    tri = mtri.Triangulation(cx_l, cy_l)
    triangles = tri.triangles  # shape (T, 3)

    # 6. Build vertex arrays — 6 * N vertices total
    all_x, all_y, all_z = [], [], []
    all_r, all_g, all_b = [], [], []
    for (_name, x_map, y_map, z_map) in faces_config:
        all_x.extend(x_map.tolist())
        all_y.extend(y_map.tolist())
        all_z.extend(z_map.tolist())
        all_r.extend(colors_float[:, 0].tolist())
        all_g.extend(colors_float[:, 1].tolist())
        all_b.extend(colors_float[:, 2].tolist())

    def _avg_color(verts):
        ri = int(sum(all_r[v] for v in verts) / len(verts) * 255)
        gi = int(sum(all_g[v] for v in verts) / len(verts) * 255)
        bi = int(sum(all_b[v] for v in verts) / len(verts) * 255)
        return '#{:02x}{:02x}{:02x}'.format(ri, gi, bi)

    faces_i, faces_j, faces_k = [], [], []
    face_colors = []

    # --- Cap faces: one filled panel per face ---
    # Even faces (0,2,4) use CCW winding; odd (1,3,5) flip to keep normals outward
    for face_idx in range(6):
        off = face_idx * N
        flip = (face_idx % 2 == 1)
        for tv in triangles:
            a, b, c = off + int(tv[0]), off + int(tv[1]), off + int(tv[2])
            color = _avg_color([a, b, c])
            if not flip:
                faces_i.append(a); faces_j.append(b); faces_k.append(c)
            else:
                faces_i.append(a); faces_j.append(c); faces_k.append(b)
            face_colors.append(color)

    # --- Bridge triangles: connect ALL face pairs through the interior ---
    # All C(6,2)=15 pairs: 3 mirror pairs + 12 adjacent pairs
    # For every unique Delaunay edge (e0,e1), create a quad strip between the two
    # faces so every face is connected to every other face.
    edge_set = set()
    for tv in triangles:
        a, b, c = int(tv[0]), int(tv[1]), int(tv[2])
        edge_set.add((min(a, b), max(a, b)))
        edge_set.add((min(b, c), max(b, c)))
        edge_set.add((min(a, c), max(a, c)))

    all_face_pairs = [(fi, fj) for fi in range(6) for fj in range(fi + 1, 6)]
    for (fi, fj) in all_face_pairs:
        off_a = fi * N
        off_b = fj * N
        for (e0, e1) in edge_set:
            va, vb = off_a + e0, off_a + e1
            wa, wb = off_b + e0, off_b + e1
            color = _avg_color([va, vb, wa])
            faces_i.append(va); faces_j.append(vb); faces_k.append(wa)
            face_colors.append(color)
            color = _avg_color([wb, vb, wa])
            faces_i.append(wb); faces_j.append(vb); faces_k.append(wa)
            face_colors.append(color)

    # --- Stage 2: Mesh Construction — shows solid 2D Delaunay fill ---
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    fig2.patch.set_facecolor('white')
    ax2.set_facecolor('white')
    for tri_verts in triangles:
        pts_x = [cx_l[tri_verts[0]], cx_l[tri_verts[1]], cx_l[tri_verts[2]]]
        pts_y = [-cy_l[tri_verts[0]], -cy_l[tri_verts[1]], -cy_l[tri_verts[2]]]
        r_avg2 = (colors_float[tri_verts[0], 0] + colors_float[tri_verts[1], 0] + colors_float[tri_verts[2], 0]) / 3
        g_avg2 = (colors_float[tri_verts[0], 1] + colors_float[tri_verts[1], 1] + colors_float[tri_verts[2], 1]) / 3
        b_avg2 = (colors_float[tri_verts[0], 2] + colors_float[tri_verts[1], 2] + colors_float[tri_verts[2], 2]) / 3
        ax2.fill(pts_x, pts_y, color=(r_avg2, g_avg2, b_avg2), linewidth=0)
    ax2.set_aspect('equal')
    ax2.axis('off')
    stages.append({
        "title": "Step 2 — Mesh Construction",
        "desc": "Delaunay triangulation connects nucleus pixels into a solid 2D mesh. Each triangle is filled with the averaged pixel color of its three vertices.",
        "img": _encode_fig(fig2, facecolor='white')
    })

    # --- Stage 3: 3D Mesh Node-Edge Graph — X/Y/Z axes, colored nodes, Delaunay edges ---
    fig3 = plt.figure(figsize=(8, 7))
    ax3 = fig3.add_subplot(111, projection='3d')
    fig3.patch.set_facecolor('white')
    ax3.set_facecolor('white')

    xs_np = np.array(all_x)
    ys_np = np.array(all_y)
    zs_np = np.array(all_z)
    node_colors_np = np.column_stack([np.array(all_r), np.array(all_g), np.array(all_b)])

    # Draw intra-face edges first (one set of Delaunay edges per face)
    for face_idx in range(6):
        off = face_idx * N
        for (e0, e1) in edge_set:
            n1, n2 = off + e0, off + e1
            ax3.plot(
                [xs_np[n1], xs_np[n2]],
                [ys_np[n1], ys_np[n2]],
                [zs_np[n1], zs_np[n2]],
                color='#7777aa', alpha=0.18, linewidth=0.4
            )

    # Draw colored nodes on top
    ax3.scatter(xs_np, ys_np, zs_np,
                c=node_colors_np, s=10, depthshade=True, zorder=5)

    # Axis labels and formatting
    ax3.set_xlabel('X', fontsize=10, labelpad=8)
    ax3.set_ylabel('Y', fontsize=10, labelpad=8)
    ax3.set_zlabel('Z', fontsize=10, labelpad=8)
    ax3.tick_params(labelsize=7)
    ax3.set_title('3D Mesh Node Graph', fontweight='bold', fontsize=12, pad=14)
    # Equal aspect ratio across all axes so cube proportions are preserved
    max_range = float(np.array([xs_np.max()-xs_np.min(),
                                 ys_np.max()-ys_np.min(),
                                 zs_np.max()-zs_np.min()]).max()) / 2.0
    mid_x = (xs_np.max() + xs_np.min()) / 2.0
    mid_y = (ys_np.max() + ys_np.min()) / 2.0
    mid_z = (zs_np.max() + zs_np.min()) / 2.0
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)
    ax3.grid(True)
    stages.append({
        "title": "Step 3 — 3D Mesh Node Graph",
        "desc": "Nucleus pixels projected onto 6 cubic faces in 3D space. Colored dots are sampled pixel nodes; gray lines are Delaunay graph edges. X/Y/Z axes show spatial extent of the cubic projection.",
        "img": _encode_fig(fig3, facecolor='white')
    })

    # 8. Build PyTorch Geometric graph — EXACTLY matching training pipeline (Model.ipynb Cell 6/10)
    #    Training used: edge records sampled at idx1%10==0, unique nodes from both endpoints,
    #    Z-score normalised X/Y/Z, RGB/255 normalised, edge_attr=[dist, avg_R_255, avg_G_255, avg_B_255]

    exact_raw_rgb = (colors_float * 255).round().astype(np.uint8)  # back to uint8 for avg computation

    img_records = []  # list of edge records, same structure as training
    for face_idx, (face_name, x_map, y_map, z_map) in enumerate(faces_config):
        for (idx1, idx2) in edge_set:
            if idx1 % 10 != 0:
                continue
            node_dist = float(np.sqrt((x_map[idx2]-x_map[idx1])**2 +
                                      (y_map[idx2]-y_map[idx1])**2 +
                                      (z_map[idx2]-z_map[idx1])**2))
            c1 = exact_raw_rgb[idx1]
            c2 = exact_raw_rgb[idx2]
            avg_rgb = ((c1.astype(int) + c2.astype(int)) // 2).astype(np.uint8)
            img_records.append({
                'face': face_name,
                'n1x': round(float(x_map[idx1]), 1), 'n1y': round(float(y_map[idx1]), 1), 'n1z': round(float(z_map[idx1]), 1),
                'n1r': int(c1[0]), 'n1g': int(c1[1]), 'n1b': int(c1[2]),
                'n2x': round(float(x_map[idx2]), 1), 'n2y': round(float(y_map[idx2]), 1), 'n2z': round(float(z_map[idx2]), 1),
                'n2r': int(c2[0]), 'n2g': int(c2[1]), 'n2b': int(c2[2]),
                'dist': round(node_dist, 2),
                'ar': int(avg_rgb[0]), 'ag': int(avg_rgb[1]), 'ab': int(avg_rgb[2]),
            })

    # Build unique node list from both endpoints (matching build_pyg_data_from_dataframe)
    def _make_key(x, y, z):
        return f"{float(x)+0.0:.1f}_{float(y)+0.0:.1f}_{float(z)+0.0:.1f}"

    seen_keys = {}
    node_xyz  = []
    node_rgb  = []
    for rec in img_records:
        for (px, py, pz, pr, pg, pb) in [
            (rec['n1x'], rec['n1y'], rec['n1z'], rec['n1r'], rec['n1g'], rec['n1b']),
            (rec['n2x'], rec['n2y'], rec['n2z'], rec['n2r'], rec['n2g'], rec['n2b']),
        ]:
            k = _make_key(px, py, pz)
            if k not in seen_keys:
                seen_keys[k] = len(node_xyz)
                node_xyz.append([px, py, pz])
                node_rgb.append([pr / 255.0, pg / 255.0, pb / 255.0])

    node_xyz_np = np.array(node_xyz, dtype=np.float32)
    node_rgb_np = np.array(node_rgb, dtype=np.float32)

    # Z-score normalise X, Y, Z (exactly as training)
    for col in range(3):
        mu  = node_xyz_np[:, col].mean()
        std = node_xyz_np[:, col].std() + 1e-6
        node_xyz_np[:, col] = (node_xyz_np[:, col] - mu) / std

    x_tensor = torch.tensor(
        np.concatenate([node_xyz_np, node_rgb_np], axis=1),
        dtype=torch.float
    )

    # Build edge index + edge attr
    src_list, dst_list, eattr_list = [], [], []
    for rec in img_records:
        k1 = _make_key(rec['n1x'], rec['n1y'], rec['n1z'])
        k2 = _make_key(rec['n2x'], rec['n2y'], rec['n2z'])
        i1, i2 = seen_keys[k1], seen_keys[k2]
        feat = [rec['dist'], float(rec['ar']), float(rec['ag']), float(rec['ab'])]
        src_list += [i1, i2];  dst_list += [i2, i1]
        eattr_list += [feat, feat]

    edge_index_tensor = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr_tensor  = torch.tensor(eattr_list, dtype=torch.float)
    batch             = torch.zeros(x_tensor.size(0), dtype=torch.long)
    data              = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)

    table_data = [
        {
            'Face': rec['face'],
            'Node1_X': rec['n1x'], 'Node1_Y': rec['n1y'], 'Node1_Z': rec['n1z'],
            'Node1_R': rec['n1r'], 'Node1_G': rec['n1g'], 'Node1_B': rec['n1b'],
            'Node2_X': rec['n2x'], 'Node2_Y': rec['n2y'], 'Node2_Z': rec['n2z'],
            'Node2_R': rec['n2r'], 'Node2_G': rec['n2g'], 'Node2_B': rec['n2b'],
            'Edge_Length': rec['dist'],
            'Line_Avg_R': rec['ar'], 'Line_Avg_G': rec['ag'], 'Line_Avg_B': rec['ab'],
        }
        for rec in img_records
    ]

    mesh_data = {
        "x": all_x, "y": all_y, "z": all_z,
        "r": all_r, "g": all_g, "b": all_b,
        "faces_i": faces_i,
        "faces_j": faces_j,
        "faces_k": faces_k,
        "face_colors": face_colors,
        "stages": stages,
        "table_data": table_data,
    }

    return data, batch, mesh_data
