import cv2


def draw_fidtm_points(frame_bgr, points, count, fps_text=None):
    out = frame_bgr.copy()

    for x, y, score in points:
        cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.circle(out, (int(x), int(y)), 7, (0, 0, 0), 1)

    cv2.rectangle(out, (20, 20), (420, 95), (0, 0, 0), -1)

    cv2.putText(
        out,
        f"FIDTM Count: {count}",
        (35, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if fps_text:
        cv2.putText(
            out,
            fps_text,
            (35, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out