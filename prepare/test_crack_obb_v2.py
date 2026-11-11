"""用合成几何检查标注转换，避免覆盖率或边界错误进入训练标签。"""
import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from prepare_crack_obb_v2 import Settings, audit_boxes, convert_binary, crack_mask, fit_box, resolve_sources, make_settings, merge_pixel_groups, pad_for_boxes


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings()

    def check_geometry(self, result, shape):
        if self.cfg.profile == 'relaxed':
            image = np.zeros((*shape, 3), np.uint8)
            _, binary, _ = pad_for_boxes(image, result['kept_mask'].copy(), result, [])
            shape = binary.shape
        self.assertGreaterEqual(audit_boxes(result['boxes'], result['kept_mask'])['coverage'], 0.995)
        for box in result['boxes']:
            self.assertTrue(np.isfinite(box).all())
            self.assertTrue((box >= -1e-5).all())
            self.assertTrue((box <= [shape[1] + 1e-5, shape[0] + 1e-5]).all())
            edges = np.roll(box, -1, axis=0) - box
            self.assertGreater(abs(cv2.contourArea(box.astype(np.float32))), 0)
            self.assertLess(abs(np.dot(edges[0], edges[1])) / max(1, np.linalg.norm(edges[0]) * np.linalg.norm(edges[1])), 1e-4)

    def test_straight_crack_stays_one_box(self):
        m = np.zeros((256, 256), np.uint8)
        cv2.line(m, (30, 40), (220, 210), 1, 5)
        result = convert_binary(m.astype(bool), self.cfg)
        self.assertEqual(len(result['boxes']), 1)
        self.check_geometry(result, m.shape)

    def test_y_junction_is_split_and_tighter(self):
        m = np.zeros((256, 256), np.uint8)
        for a, b in [((128, 125), (30, 35)), ((128, 125), (225, 35)), ((128, 125), (128, 235))]:
            cv2.line(m, a, b, 1, 7)
        result = convert_binary(m.astype(bool), self.cfg)
        self.assertGreaterEqual(len(result['boxes']), 3)
        yy, xx = np.nonzero(m)
        old, _ = fit_box(np.c_[xx, yy], m.shape, 2)
        new_area = audit_boxes(result['boxes'], result['kept_mask'])['box_area_sum']
        self.assertLess(new_area, cv2.contourArea(old.astype(np.float32)) * 0.55)
        self.check_geometry(result, m.shape)

    def test_curved_crack_is_split(self):
        m = np.zeros((256, 256), np.uint8)
        cv2.ellipse(m, (128, 128), (90, 85), 0, 10, 170, 1, 5)
        result = convert_binary(m.astype(bool), self.cfg)
        self.assertGreater(len(result['boxes']), 2)
        self.check_geometry(result, m.shape)

    def test_red_spalling_is_not_crack(self):
        m = np.zeros((64, 64, 3), np.uint8)
        m[10:20, 10:20] = (0, 0, 128)
        result = convert_binary(crack_mask(m), self.cfg)
        self.assertEqual(result['boxes'], [])
        self.assertEqual(result['kept_mask'].sum(), 0)

    def test_boundary_box_is_rectangular_and_covers_pixels(self):
        m = np.zeros((128, 128), np.uint8)
        cv2.line(m, (0, 5), (115, 110), 1, 7)
        result = convert_binary(m.astype(bool), self.cfg)
        self.check_geometry(result, m.shape)

    def test_nearby_components_remain_separate(self):
        m = np.zeros((128, 128), np.uint8)
        cv2.line(m, (10, 60), (55, 60), 1, 3)
        cv2.line(m, (65, 60), (115, 60), 1, 3)
        result = convert_binary(m.astype(bool), self.cfg)
        self.assertEqual(result['components'], 2)
        self.assertEqual(len(result['boxes']), 2)
        self.check_geometry(result, m.shape)

    def test_short_spurs_do_not_fragment_a_straight_trunk(self):
        m = np.zeros((256, 256), np.uint8)
        cv2.line(m, (25, 128), (230, 128), 1, 7)
        for x in range(45, 220, 25):
            cv2.line(m, (x, 128), (x, 118), 1, 3)
        result = convert_binary(m.astype(bool), self.cfg)
        self.assertLessEqual(len(result['boxes']), 2)
        self.check_geometry(result, m.shape)

    def test_relocated_sources_keep_split_and_do_not_mutate_original(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            for folder, suffix in [('outputs_RGB', '.jpg'), ('outputs_Mask', '.png')]:
                p = root / 'labelme' / folder / ('example' + suffix)
                p.parent.mkdir(parents=True)
                p.write_bytes(b'test')
            row = {'sample_id': 'example', 'split': 'val', 'source_group': 'source-A',
                   'image_path': 'old/labelme/outputs_RGB/example.jpg',
                   'mask_path': 'old/labelme/outputs_Mask/example.png'}
            converted = resolve_sources([row], root)[0]
            self.assertEqual(converted['split'], 'val')
            self.assertEqual(converted['source_group'], 'source-A')
            self.assertTrue(Path(converted['mask_path']).is_file())
            self.assertEqual(row['mask_path'], 'old/labelme/outputs_Mask/example.png')

    def test_missing_sources_fail_before_conversion(self):
        with self.assertRaises(FileNotFoundError):
            resolve_sources([{'image_path': 'missing-photo.jpg', 'mask_path': 'missing-mask.png'}], None)


class RelaxedTests(ConversionTests):
    """沿用基础覆盖/颜色/分组检查，再检查碎框合并不破坏真实分叉。"""

    def setUp(self):
        self.cfg = make_settings('relaxed')

    def test_y_junction_is_split_and_tighter(self):
        m = np.zeros((256, 256), np.uint8)
        for a, b in [((128, 125), (30, 35)), ((128, 125), (225, 35)), ((128, 125), (128, 235))]:
            cv2.line(m, a, b, 1, 7)
        result = convert_binary(m.astype(bool), self.cfg)
        self.assertGreaterEqual(len(result['boxes']), 3)
        self.assertLessEqual(len(result['boxes']), 4)
        self.check_geometry(result, m.shape)

    def test_curved_crack_is_split(self):
        m = np.zeros((256, 256), np.uint8)
        cv2.ellipse(m, (128, 128), (90, 85), 0, 10, 170, 1, 5)
        result = convert_binary(m.astype(bool), self.cfg)
        self.assertGreaterEqual(len(result['boxes']), 2)
        self.assertLessEqual(len(result['boxes']), 4)
        self.check_geometry(result, m.shape)

    def test_collinear_segments_merge_without_discarding_pixels(self):
        y, x = np.mgrid[60:64, 20:180]
        pixels = np.c_[x.ravel(), y.ravel()]
        groups, count = merge_pixel_groups([pixels[pixels[:, 0] < 80], pixels[pixels[:, 0] >= 80]], self.cfg)
        self.assertEqual(count, 1)
        self.assertEqual(len(groups), 1)
        self.assertEqual({tuple(p) for p in groups[0]}, {tuple(p) for p in pixels})

    def test_gentle_wiggles_do_not_create_arbitrary_cut_points(self):
        m = np.zeros((256, 512), np.uint8)
        x = np.arange(12, 500)
        p = np.c_[x, 128 + 6 * np.sin(x / 25)].astype(np.int32)
        cv2.polylines(m, [p], False, 1, 3)
        result = convert_binary(m.astype(bool), self.cfg)
        self.assertEqual(len(result['boxes']), 1)
        self.check_geometry(result, m.shape)

    def test_boundary_does_not_generate_tiny_extra_targets(self):
        m = np.zeros((512, 512), np.uint8)
        cv2.line(m, (0, 8), (511, 450), 1, 3)
        result = convert_binary(m.astype(bool), self.cfg)
        self.check_geometry(result, m.shape)
        lengths = [max(np.linalg.norm(b[1]-b[0]), np.linalg.norm(b[2]-b[1])) for b in result['boxes']]
        self.assertGreater(min(lengths), 32)
        self.assertLessEqual(len(result['boxes']), 6)

    def test_real_small_component_is_kept(self):
        m = np.zeros((64, 64), bool)
        m[20:23, 20:25] = True
        result = convert_binary(m, self.cfg)
        self.assertEqual(len(result['boxes']), 1)
        self.assertEqual(result['kept_mask'].sum(), 15)
        self.check_geometry(result, m.shape)

    def test_small_direction_change_keeps_one_major_segment(self):
        m = np.zeros((256, 512), np.uint8)
        cv2.polylines(m, [np.array([[20, 80], [220, 80], [420, 130]], np.int32)], False, 1, 5)
        result = convert_binary(m.astype(bool), self.cfg)
        self.assertEqual(len(result['boxes']), 1)
        self.check_geometry(result, m.shape)

    def test_large_angle_splits_visible_legs(self):
        m = np.zeros((300, 300), np.uint8)
        cv2.polylines(m, [np.array([[20, 30], [230, 30], [230, 240]], np.int32)], False, 1, 5)
        result = convert_binary(m.astype(bool), self.cfg)
        self.assertEqual(len(result['boxes']), 2)
        self.check_geometry(result, m.shape)

    def test_border_diagonal_keeps_its_orientation_with_padding(self):
        m = np.zeros((256, 256), np.uint8)
        cv2.line(m, (0, 50), (250, 200), 1, 5)
        result = convert_binary(m.astype(bool), self.cfg)
        image = np.full((256, 256, 3), 80, np.uint8)
        original = image.copy()
        padded, binary, padding = pad_for_boxes(image, m.astype(bool), result, [])
        self.assertEqual(len(result['boxes']), 1)
        box = result['boxes'][0]
        edges = np.roll(box, -1, axis=0) - box
        vector = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        angle = abs(np.degrees(np.arctan2(vector[1], vector[0]))) % 180
        angle = min(angle, 180-angle)
        self.assertGreater(angle, 20)
        self.assertLess(angle, 40)
        left, top, _, _ = padding
        np.testing.assert_array_equal(padded[top:top+256, left:left+256], original)
        self.assertEqual(audit_boxes(result['boxes'], result['kept_mask'])['coverage'], 1.0)
        self.assertTrue((box >= 0).all())
        self.assertTrue((box <= [binary.shape[1], binary.shape[0]]).all())


if __name__ == '__main__':
    unittest.main()
