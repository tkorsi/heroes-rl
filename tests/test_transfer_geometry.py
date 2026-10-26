"""Regression for the authored anchor-versus-visitable fixture mistake."""
import unittest
from tools.validate_transfer_geometry import validate


class NativeGeometryControls(unittest.TestCase):
    def test_anchor_on_board_can_have_off_board_native_visitable(self):
        result=validate([0,7],[4,2])
        self.assertFalse(result['valid']);self.assertEqual(result['hero_visitable'],[-1,7])

    def test_left_edge_visitable_requires_hero_anchor_one(self):
        result=validate([1,6],[7,6])
        self.assertTrue(result['valid']);self.assertEqual(result['hero_visitable'],[0,6])

    def test_collision_uses_native_visitable_not_sprite_anchor(self):
        self.assertFalse(validate([5,5],[7,6])['valid'])
        self.assertTrue(validate([6,6],[4,6])['valid'])


if __name__=='__main__':unittest.main()
