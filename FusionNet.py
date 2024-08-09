import torch
import torch.nn as nn
from MultiModal.EnhancedSemanticAttentionModule import EnhancedSemanticAttentionModule
from MultiModal.EnhancedSemanticAttentionModule2 import EnhancedSemanticAttentionModule2
from MultiModal.EnhancedSemanticAttentionModule3 import EnhancedSemanticAttentionModule3
from MultiModal.FeatureFusionModule import FeatureFusionModule
from MultiModal.FeatureFusionModule2 import FeatureFusionModule2
from MultiModal.FeatureFusionModule3 import FeatureFusionModule3
from MultiModal.FeatureFusionModule4 import FeatureFusionModule4
from MultiModal.ModalFusionModule import ModalFusionModule
from MultiModal.ModalFusionModule2 import ModalFusionModule2
from MultiModal.ModalFusionModule3 import ModalFusionModule3
from MultiModal.ModalFusionModule4 import ModalFusionModule4
from MultiModal.ModalFusionModule5 import ModalFusionModule5
from MultiModal.ModalFusionModule6 import ModalFusionModule6
from MultiModal.ModalFusionModule7 import ModalFusionModule7
from MultiModal.ModalFusionModule8 import ModalFusionModule8
from MultiModal.ModalFusionModule9 import ModalFusionModule9
from MultiModal.ModalFusionModule10 import ModalFusionModule10
from MultiModal.ModalFusionModule11 import ModalFusionModule11
from MultiModal.ModalFusionModule12 import ModalFusionModule12
from MultiModal.ModalFusionModule13 import ModalFusionModule13
from MultiModal.ConditionalFusionModule14 import ConditionalFusionModule14
from MultiModal.QueryEnhancedSemanticModule import QueryEnhancedSemanticModule
from MultiModal.SemanticAttentionModule import SemanticAttentionModule
from MultiModal.SemanticAttentionModule2 import SemanticAttentionModule2


# 定义一个名为 FusionNet 的神经网络类，它继承自 nn.Module
class FusionNet(nn.Module):
    def __init__(self, choice, global_dim=None, local_dim=None):
        self.global_dim = global_dim  # 设置全局输入的维度
        self.local_dim = local_dim  # 设置局部输入的维度
        super(FusionNet, self).__init__()
        self.choice = choice  # 保存选择的模块索引

        # 根据 choice 选择不同的模块进行初始化
        if choice == 0:
            self.block = EnhancedSemanticAttentionModule(global_dim=self.global_dim, local_dim=self.local_dim,
                                                         num_heads=8).cuda()
        if choice == 1:
            self.block = EnhancedSemanticAttentionModule2(global_dim=self.global_dim, local_dim=self.local_dim,
                                                          num_heads=8).cuda()
        if choice == 2:
            self.block = EnhancedSemanticAttentionModule3(global_dim=self.global_dim, local_dim=self.local_dim,
                                                          num_heads=8).cuda()
        if choice == 3:
            self.block = FeatureFusionModule(global_dim=self.global_dim, local_dim=self.local_dim).cuda()
        if choice == 4:
            self.block = FeatureFusionModule2(global_dim=self.global_dim, local_dim=self.local_dim).cuda()
        if choice == 5:
            self.block = FeatureFusionModule3(global_dim=self.global_dim, local_dim=self.local_dim).cuda()
        if choice == 6:
            self.block = FeatureFusionModule4(global_dim=self.global_dim, local_dim=self.local_dim).cuda()
        if choice == 7:
            self.block = ModalFusionModule(global_dim=self.global_dim, local_dim=self.local_dim, num_heads=8).cuda()
        if choice == 8:
            self.block = ModalFusionModule2(global_dim=self.global_dim, local_dim=self.local_dim, num_heads=8).cuda()
        if choice == 9:
            self.block = ModalFusionModule3(global_dim=self.global_dim, local_dim=self.local_dim, num_heads=8).cuda()
        if choice == 10:
            self.block = ModalFusionModule4(global_dim=self.global_dim, local_dim=self.local_dim, num_heads=8).cuda()
        if choice == 11:
            self.block = ModalFusionModule5(global_dim=self.global_dim, local_dim=self.local_dim, num_heads=8).cuda()
        if choice == 12:
            self.block = ModalFusionModule6(global_dim=self.global_dim, local_dim=self.local_dim, num_heads=8).cuda()
        if choice == 13:
            self.block = ModalFusionModule7(global_dim=self.global_dim, local_dim=self.local_dim, num_heads=8).cuda()
        if choice == 14:
            self.block = ModalFusionModule8(global_dim=self.global_dim, local_dim=self.local_dim, num_heads=8).cuda()
        if choice == 15:
            self.block = ModalFusionModule9(global_dim=self.global_dim, local_dim=self.local_dim).cuda()
        if choice == 16:
            self.block = ModalFusionModule10(global_dim=self.global_dim, local_dim=self.local_dim).cuda()
        if choice == 17:
            self.block = ModalFusionModule11(global_dim=self.global_dim, local_dim=self.local_dim, num_heads=8,
                                             num_encoder_layers=2).cuda()
        if choice == 18:
            self.block = ModalFusionModule12(global_dim=self.global_dim, local_dim=self.local_dim).cuda()
        if choice == 19:
            self.block = ModalFusionModule13(global_dim=self.global_dim, local_dim=self.local_dim).cuda()
        if choice == 20:
            self.block = ConditionalFusionModule14(global_dim=self.global_dim, local_dim=self.local_dim).cuda()
        if choice == 21:
            self.block = QueryEnhancedSemanticModule(global_dim=self.global_dim, local_dim=self.local_dim,
                                                     num_heads=8).cuda()
        if choice == 22:
            self.block = SemanticAttentionModule(global_dim=self.global_dim, local_dim=self.local_dim,
                                                 num_heads=8).cuda()
        if choice == 23:
            self.block = SemanticAttentionModule2(global_dim=self.global_dim, local_dim=self.local_dim,
                                                  num_heads=8).cuda()

    # 定义前向传播函数
    def forward(self, global_input, local_input):
        # 将输入传递给选择的模块并返回输出
        output = self.block(global_input, local_input)
        return output


if __name__ == '__main__':
    # 示例使用
    global_input = torch.rand(2, 33, 256).cuda()  # 示例的全局特征输入
    local_input = torch.rand(2, 33, 512).cuda()  # 示例的局部特征输入（维度不同）

    # 依次使用不同的模块进行测试
    for i in range(24):
        block = FusionNet(choice=i, global_dim=256, local_dim=512).cuda()  # 根据 choice 选择模块
        output = block(global_input, local_input)
        print(i, "    ", global_input.size(), local_input.size(), output.size())
