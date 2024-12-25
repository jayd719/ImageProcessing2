def process_objects(objects_directory, stitched_image):
    pano_kp, pano_des = detect_and_describe(stitched_image)
    output_path = os.path.join("./Panorama", "Panorama_bb.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for i, objectImageFile in enumerate(sorted(os.listdir(objects_directory))):
        objectImage = cv.imread(os.path.join(objects_directory, objectImageFile))
        obj_kp, obj_des = detect_and_describe(objectImage)

        matches = match_keypoints(obj_des, pano_des)
        if len(matches) < 10:
            continue

        homography, _ = compute_homography(obj_kp, pano_kp, matches)
        draw_matches(i, objectImage, stitched_image, obj_kp, pano_kp, matches)

        matches_coords = np.array([pano_kp[match.trainIdx].pt for match in matches])

        clustering = DBSCAN(eps=30, min_samples=3).fit(matches_coords)
        labels = clustering.labels_

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        largest_cluster_id = max(
            unique_labels, key=lambda label: list(labels).count(label)
        )

        cluster_points = matches_coords[labels == largest_cluster_id]
        x, y, x_max, y_max = calculate_bounding_box(
            cluster_points, object_names[i], stitched_image.shape
        )

        color = (randint(1, 255), randint(1, 255), randint(1, 255))
        stitched_image = cv.rectangle(
            stitched_image, (x, y), (x_max, y_max), color, 5, cv.LINE_AA
        )
        cv.putText(
            stitched_image,
            object_names[i],
            (x + 20, y + 30),
            cv.FONT_HERSHEY_TRIPLEX,
            1.25,
            (0, 0, 0),
            3,
        )

    cv.imwrite(output_path, stitched_image)
    print("Panorama_bb created successfully:", output_path)
