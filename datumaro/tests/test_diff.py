from unittest import TestCase

from datumaro.components.extractor import DatasetItem, Label, Bbox
from datumaro.components.operations import DistanceComparator, ExactComparator


class DistanceComparatorTest(TestCase):
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

# class ExactComparatorTest(TestCase):
#     def test_





#         label_categories = LabelCategories()
#         for i in range(5):
#             label_categories.add('cat' + str(i))

#         mask_categories = MaskCategories(
#             generate_colormap(len(label_categories.items)))

#         points_categories = PointsCategories()
#         for index, _ in enumerate(label_categories.items):
#             points_categories.add(index, ['cat1', 'cat2'], joints=[[0, 1]])

#         return Dataset.from_iterable([
#             DatasetItem(id=100, subset='train', image=np.ones((10, 6, 3)),
#                 annotations=[
#                     Caption('hello', id=1),
#                     Caption('world', id=2, group=5),
#                     Label(2, id=3, attributes={
#                         'x': 1,
#                         'y': '2',
#                     }),
#                     Bbox(1, 2, 3, 4, label=4, id=4, z_order=1, attributes={
#                         'score': 1.0,
#                     }),
#                     Bbox(5, 6, 7, 8, id=5, group=5),
#                     Points([1, 2, 2, 0, 1, 1], label=0, id=5, z_order=4),
#                     Mask(label=3, id=5, z_order=2, image=np.ones((2, 3))),
#                 ]),
#             DatasetItem(id=21, subset='train',
#                 annotations=[
#                     Caption('test'),
#                     Label(2),
#                     Bbox(1, 2, 3, 4, label=5, id=42, group=42)
#                 ]),

#             DatasetItem(id=2, subset='val',
#                 annotations=[
#                     PolyLine([1, 2, 3, 4, 5, 6, 7, 8], id=11, z_order=1),
#                     Polygon([1, 2, 3, 4, 5, 6, 7, 8], id=12, z_order=4),
#                 ]),

#             DatasetItem(id=42, subset='test',
#                 attributes={'a1': 5, 'a2': '42'}),

#             DatasetItem(id=42),
#             DatasetItem(id=43, image=Image(path='1/b/c.qq', size=(2, 4))),
#         ], categories={
#             AnnotationType.label: label_categories,
#             AnnotationType.mask: mask_categories,
#             AnnotationType.points: points_categories,
#         })