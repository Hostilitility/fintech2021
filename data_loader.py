# 数据读取。可以设定模式 (prediction，validation)，
#          并根据模式读取相应的训练数据（train）或测试数据（test）。


from utils import *


class DataLoader(object):
    def __init__(self, val=False):
        """
        当 val 为 True 时，为【validation模式】。
        此时： load_train 将返回 18 年的数据训练
              load_test 将返回 19 年的数据测试
        当 val 为 False 时，为【prediction模式】。
        此时： load_train 将返回 18 + 19 年的数据训练
              load_test 将返回 20 年的数据测试

        :param val: (boolean)
        """
        self.val = val
        # TODO: 数据路径，需要自行设置👇
        self.file_path = "data/data0.csv"
        self.data = load_item(self.file_path)

    def load_train(self) -> pd.DataFrame:
        if self.val:
            return self.data[self.data['year'] == 2018]
        else:
            return self.data[self.data['year'] != 2020]

    def load_test(self) -> pd.DataFrame:
        if self.val:
            return self.data[self.data['year'] == 2019]
        else:
            test_data = self.data[self.data['year'] == 2020]
            # assert len(test_data) == 25293
            return test_data
