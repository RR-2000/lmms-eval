import io
import logging
import os
import re
from collections import defaultdict

from PIL import Image
import numpy as np

eval_logger = logging.getLogger("lmms-eval")


CATEGORY_ALIASES = {
    "Positional Relationship (Obj.-Obj.)": "Positional Relationship (Obj.–Obj.)",
    "Positional Relationship (Cam.-Obj.)": "Positional Relationship (Cam.–Obj.)",
    "Positional Relationship (Cam.-Cam.)": "Positional Relationship (Cam.–Cam.)",
    "Positional Relationship (Obj.-Reg.)": "Positional Relationship (Obj.–Reg.)",
    "Positional Relationship (Cam.-Reg.)": "Positional Relationship (Cam.–Reg.)",
    "Positional Relationship (Reg.-Reg.)": "Positional Relationship (Reg.–Reg.)",
}


def msr_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if not isinstance(lmms_eval_specific_kwargs, dict):
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc["question"].strip()
    if pre_prompt != "":
        question = f"{pre_prompt}{question}"
    if post_prompt != "":
        question = f"{question}{post_prompt}"
    return question

def msr_doc_to_visual(doc):
    print(f"Extracting images from doc with id: {doc['id']}")
    image_list = []
    for img_data in doc["images"]:
        if isinstance(img_data, Image.Image):
            image = img_data.convert("RGB")
        else:
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
        image_list.append(image)

    CORR = os.getenv("MMSI_BENCH_DRAW_CORR", "0").lower() in ("1", "true", "yes")    
    # if True:
    print(f"Drawing ORB correspondences for {len(image_list)} images in doc {doc['id']}")
    image_list = draw_orb_correspondences(image_list)
    return image_list


def draw_orb_correspondences(
    images,
    grid_size: int = 15,
    max_matches: int = 60,
    use_ransac: bool = False,
):
    try:
        import cv2
    except Exception as exc:
        eval_logger.warning("OpenCV unavailable for correspondence drawing: %s", exc)
        print(f"[corr] OpenCV unavailable: {exc}")
        return images

    if len(images) < 2:
        print("[corr] Need at least 2 images; skipping")
        return images

    np_images = [np.array(image.convert("RGB")) for image in images]
    print(f"[corr] Converted {len(np_images)} images to NumPy arrays")
    regions_by_image = []
    centers_by_image = []
    descriptors_by_image = []
    keypoints_by_image = []

    orb = cv2.ORB_create()
    for idx, np_img in enumerate(np_images):
        regions = _grid_regions(np_img.shape[1], np_img.shape[0], grid_size)
        keypoints, descriptors = orb.detectAndCompute(np_img, None)
        print(
            f"[corr] Image {idx}: keypoints={0 if keypoints is None else len(keypoints)}, "
            f"descriptors={'none' if descriptors is None else len(descriptors)}"
        )
        region_desc, region_centers, region_kps = _region_descriptors(
            regions, keypoints, descriptors
        )
        print(
            f"[corr] Image {idx}: region_desc={'none' if region_desc is None else len(region_desc)}"
        )
        regions_by_image.append(regions)
        centers_by_image.append(region_centers)
        descriptors_by_image.append(region_desc)
        keypoints_by_image.append(region_kps)

    reference_desc = descriptors_by_image[0]
    reference_centers = centers_by_image[0]
    reference_kps = keypoints_by_image[0]
    if reference_desc is None or len(reference_desc) == 0:
        print("[corr] Reference image has no descriptors; skipping")
        return images

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    colors = _color_palette(max_matches)
    output_images = [img.copy() for img in np_images]

    for idx in range(1, len(np_images)):
        desc = descriptors_by_image[idx]
        centers = centers_by_image[idx]
        kps = keypoints_by_image[idx]
        if desc is None or len(desc) == 0:
            print(f"[corr] Image {idx}: no descriptors to match")
            continue

        matches = bf.match(reference_desc, desc)
        matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
        inlier_matches = _spatially_verify_matches(
            reference_kps,
            kps,
            matches,
            use_ransac=use_ransac,
        )
        print(f"[corr] Image {idx}: {len(matches)} matches, {len(inlier_matches)} inliers")

        for color_idx, match in enumerate(inlier_matches):
            ref_center = reference_centers[match.queryIdx]
            tgt_center = centers[match.trainIdx]
            color = colors[color_idx % len(colors)]
            cv2.circle(output_images[0], ref_center, 50, color, -1)
            cv2.circle(output_images[idx], tgt_center, 50, color, -1)

    return [Image.fromarray(img).convert("RGB") for img in output_images]


def _grid_regions(width: int, height: int, grid_size: int):
    regions = []
    cell_w = max(1, width // grid_size)
    cell_h = max(1, height // grid_size)
    for row in range(grid_size):
        for col in range(grid_size):
            x0 = col * cell_w
            y0 = row * cell_h
            x1 = width if col == grid_size - 1 else (col + 1) * cell_w
            y1 = height if row == grid_size - 1 else (row + 1) * cell_h
            regions.append((x0, y0, x1, y1))
    return regions


def _region_descriptors(regions, keypoints, descriptors):
    if not keypoints or descriptors is None:
        return None, [], []

    region_best = [None] * len(regions)
    region_kps = [None] * len(regions)
    for kp, desc in zip(keypoints, descriptors):
        x, y = kp.pt
        for idx, (x0, y0, x1, y1) in enumerate(regions):
            if x0 <= x < x1 and y0 <= y < y1:
                current = region_kps[idx]
                if current is None or kp.response > current.response:
                    region_kps[idx] = kp
                    region_best[idx] = desc
                break

    region_desc = []
    region_centers = []
    region_kps_filtered = []
    for idx, desc in enumerate(region_best):
        if desc is None:
            continue
        x0, y0, x1, y1 = regions[idx]
        center = (int((x0 + x1) / 2), int((y0 + y1) / 2))
        region_centers.append(center)
        region_desc.append(desc)
        region_kps_filtered.append(region_kps[idx])

    if not region_desc:
        return None, [], []

    return np.stack(region_desc), region_centers, region_kps_filtered


def _spatially_verify_matches(ref_kps, tgt_kps, matches, use_ransac: bool = True):
    try:
        import cv2
    except Exception:
        return matches

    if not use_ransac:
        return matches

    if len(matches) < 4:
        return matches

    ref_pts = np.float32([ref_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    tgt_pts = np.float32([tgt_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(ref_pts, tgt_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return matches

    inliers = []
    for match, flag in zip(matches, mask.ravel().tolist()):
        if flag:
            inliers.append(match)
    return inliers


def _color_palette(count: int):
    colors = []
    for idx in range(count):
        hue = idx / max(1, count)
        r, g, b = _hsv_to_rgb(hue, 0.9, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


def _hsv_to_rgb(h, s, v):
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def extract_single_choice_with_word_boundary(pred, gt):
    if pred is None:
        return None

    pred = str(pred)

    pattern_1 = r"``([^`]*)``"
    match = re.search(pattern_1, pred)
    if match:
        pred = match.group(1)

    pattern_2 = r"`([^`]*)`"
    match = re.search(pattern_2, pred)
    if match:
        pred = match.group(1)

    pattern_add = r"\{([^}]*)\}"
    match = re.search(pattern_add, pred)
    if match:
        pred = match.group(1)

    pattern_3 = r"\b[A-F]\b(?!\s[a-zA-Z])"
    match = re.search(pattern_3, pred, flags=re.IGNORECASE)
    if match:
        pred = match.group().upper()
    else:
        return None

    answer = gt.lower().replace("\n", " ").strip()
    predict = pred.lower().replace("\n", " ").strip()
    try:
        if answer == predict[0]:
            return 1.0
        elif predict[0] == "(" and answer == predict[1]:
            return 1.0
        elif predict[0:7] == "option " and answer == predict[7]:
            return 1.0
        elif predict[0:14] == "the answer is " and answer == predict[14]:
            return 1.0
    except Exception:
        return 0.0
    return 0.0


def msr_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0]
    gt = doc["answer"]

    score = extract_single_choice_with_word_boundary(pred, gt)
    category = CATEGORY_ALIASES.get(doc["question_type"], doc["question_type"])
    l2_category = category
    if score is None:
        return {category: {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "note": "cannot find answer"}, "average": {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "note": "cannot find answer"}}
    return {category: {"question_id": doc["id"], "l2_category": l2_category, "score": score}, "average": {"question_id": doc["id"], "l2_category": l2_category, "score": score}}


def msr_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    l2_category_scores = defaultdict(list)
    for result in results:
        score = result["score"]
        l2_category = result["l2_category"]
        l2_category_scores[l2_category].append(score)

    l2_category_avg_score = {}
    for l2_category, scores in l2_category_scores.items():
        avg_score = sum(scores) / len(scores)
        l2_category_avg_score[l2_category] = avg_score
        eval_logger.info(f"{l2_category}: {avg_score:.2f}")

    all_scores = [score for scores in l2_category_scores.values() for score in scores]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return avg_score


def _debug_orb_example() -> None:
    try:
        import cv2
    except Exception as exc:
        print(f"[corr] OpenCV unavailable: {exc}")
        return

    width, height = 640, 480
    base = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(base, (80, 120), (240, 300), (255, 255, 255), -1)
    cv2.circle(base, (420, 200), 50, (255, 255, 255), -1)
    cv2.line(base, (300, 380), (540, 380), (255, 255, 255), 5)

    shifted = np.zeros_like(base)
    dx, dy = 40, -20
    shifted[max(0, dy):height + min(0, dy), max(0, dx):width + min(0, dx)] = base[
        max(0, -dy):height - max(0, dy),
        max(0, -dx):width - max(0, dx),
    ]

    images = [Image.fromarray(base), Image.fromarray(shifted)]
    output_images = draw_orb_correspondences(images, grid_size=10, max_matches=80, use_ransac=False)
    for idx, image in enumerate(output_images):
        output_path = f"orb_debug_{idx}.png"
        image.save(output_path)
        print(f"[corr] Saved {output_path}")


if __name__ == "__main__":
    _debug_orb_example()
