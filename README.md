# AdvAttackPractice

> 比赛参考[2024羊城杯粤港澳大湾区网络安全大赛本科组](https://github.com/CTF-Archives/2024-YCB-Undergraduate) 
>
> 参赛队伍：Del0n1x；ranking：23/500（竞争太激烈力，打麻了😭）
>
> 队伍[writeup](https://www.yuque.com/keyboard-ovrmx/scxvuu/rckx59lv20a3h6uy?singleDoc# 《2024 羊城杯》)已公开访问，欢迎师傅们参考交流

## NLP_Model_Attack

> 题目分值：已答出45次，初始分值500.0，当前分值483.2，解出分值482.43 题目难度：中等

使用[TextAttack](https://github.com/QData/TextAttack)工具箱，详细教程和代码示例参考[TextAttack Documentation — TextAttack 0.3.10 documentation](https://textattack.readthedocs.io/en/master/)，各类API文档使用方法写得很好，writeup也可以写得很简洁。

## Targeted_Image_adv_attacks

> 题目分值：已答出10次，初始分值500.0，当前分值499.3，解出分值499.13 题目难度：困难

使用[torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch)工具箱，好用爱用，

重写数据加载方法，PGD CW策略参数调试，150张图片改写批量运行，三类图片分别攻击，运气也很好PGD策略能本地过测，最后提交多一点点过关。

结果通过Nvidia RTX2080TI 本地计算Average SSIM: 0.9547 Average Attack Success Rate: 0.9200

经过优化能在17s左右生成150张对抗样本且满足上述数据效果，代码是经过claude-3.5-sonnet简化的版本，环境参考[AdvAttack requirements](https://github.com/Harry24k/adversarial-attacks-pytorch)。

