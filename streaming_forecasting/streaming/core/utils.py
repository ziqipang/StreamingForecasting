from .bank import TrajBank
from ..macros import AGT_OBJ_CLASSES


def seq2tracklets(bboxes, ids, classes, selected_classes=AGT_OBJ_CLASSES):
    # get the tracklets
    bank = TrajBank()
    for _, (frame_bboxes, frame_ids, frame_classes) in enumerate(zip(bboxes, ids, classes)):
        if selected_classes is not None:
            selected_idx = [i for i, c in enumerate(frame_classes) if c in selected_classes]
            frame_bboxes = [frame_bboxes[i] for i in selected_idx]
            frame_ids = [frame_ids[i] for i in selected_idx]
            frame_classes = [frame_classes[i] for i in selected_idx]

        bank.update(frame_bboxes, frame_ids, frame_classes)
        bank.frame_index += 1
    bank.frame_index -= 1
    tracklets = bank.get_all_tracklets()
    return tracklets