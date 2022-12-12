import unittest
from common import Item, DataVerify


class TestLib(unittest.TestCase):

    def test_pre_check(self):
        # item_mock = mock()

        values = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
                  [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
                  [1, 1, 1, 1]]
        # values= [[1, 1, 1, 1]]
        item = Item(meeting_id='test', name='test', start_timestamp="0000000000", end_timestamp="0000000000",
                    values=values)
        self.assertTrue(DataVerify.pre_check_data(item))
        # self.assertFalse(data_verify.pre_check_data(item))


if __name__ == '__main__':
    unittest.main()
