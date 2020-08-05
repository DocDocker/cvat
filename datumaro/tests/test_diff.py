from unittest import TestCase

from datumaro.components.extractor import DatasetItem, Label, Bbox
from datumaro.components.operations import DistanceComparator


class DiffTest(TestCase):
    def test_no_bbox_diff_with_same_item(self):
        detections = 3
        anns = [
            Bbox(i * 10, 10, 10, 10, label=i)
            for i in range(detections)
        ]
        item = DatasetItem(id=0, annotations=anns)

        iou_thresh = 0.5
        comp = DistanceComparator(iou_threshold=iou_thresh)

        result = comp.match_boxes(item, item)

        matches, mispred, a_greater, b_greater = result
        self.assertEqual(0, len(mispred))
        self.assertEqual(0, len(a_greater))
        self.assertEqual(0, len(b_greater))
        self.assertEqual(len(item.annotations), len(matches))
        for a_bbox, b_bbox in matches:
            self.assertLess(iou_thresh, a_bbox.iou(b_bbox))
            self.assertEqual(a_bbox.label, b_bbox.label)

    def test_can_find_bbox_with_wrong_label(self):
        detections = 3
        class_count = 2
        item1 = DatasetItem(id=1, annotations=[
            Bbox(i * 10, 10, 10, 10, label=i)
            for i in range(detections)
        ])
        item2 = DatasetItem(id=2, annotations=[
            Bbox(i * 10, 10, 10, 10, label=(i + 1) % class_count)
            for i in range(detections)
        ])

        iou_thresh = 0.5
        comp = DistanceComparator(iou_threshold=iou_thresh)

        result = comp.match_boxes(item1, item2)

        matches, mispred, a_greater, b_greater = result
        self.assertEqual(len(item1.annotations), len(mispred))
        self.assertEqual(0, len(a_greater))
        self.assertEqual(0, len(b_greater))
        self.assertEqual(0, len(matches))
        for a_bbox, b_bbox in mispred:
            self.assertLess(iou_thresh, a_bbox.iou(b_bbox))
            self.assertEqual((a_bbox.label + 1) % class_count, b_bbox.label)

    def test_can_find_missing_boxes(self):
        detections = 3
        class_count = 2
        item1 = DatasetItem(id=1, annotations=[
            Bbox(i * 10, 10, 10, 10, label=i)
            for i in range(detections) if i % 2 == 0
        ])
        item2 = DatasetItem(id=2, annotations=[
            Bbox(i * 10, 10, 10, 10, label=(i + 1) % class_count)
            for i in range(detections) if i % 2 == 1
        ])

        iou_thresh = 0.5
        comp = DistanceComparator(iou_threshold=iou_thresh)

        result = comp.match_boxes(item1, item2)

        matches, mispred, a_greater, b_greater = result
        self.assertEqual(0, len(mispred))
        self.assertEqual(len(item1.annotations), len(a_greater))
        self.assertEqual(len(item2.annotations), len(b_greater))
        self.assertEqual(0, len(matches))

    def test_no_label_diff_with_same_item(self):
        detections = 3
        anns = [
            Label(i) for i in range(detections)
        ]
        item = DatasetItem(id=1, annotations=anns)

        result = DistanceComparator().match_labels(item, item)

        matches, a_greater, b_greater = result
        self.assertEqual(0, len(a_greater))
        self.assertEqual(0, len(b_greater))
        self.assertEqual(len(item.annotations), len(matches))

    def test_can_find_wrong_label(self):
        item1 = DatasetItem(id=1, annotations=[
            Label(0),
            Label(1),
            Label(2),
        ])
        item2 = DatasetItem(id=2, annotations=[
            Label(2),
            Label(3),
            Label(4),
        ])

        result = DistanceComparator().match_labels(item1, item2)

        matches, a_greater, b_greater = result
        self.assertEqual(2, len(a_greater))
        self.assertEqual(2, len(b_greater))
        self.assertEqual(1, len(matches))