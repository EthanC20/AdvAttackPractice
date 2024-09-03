# AdvAttackPractice

> 比赛参考2024羊城杯粤港澳大湾区网络安全大赛本科组[2024-YCB-Undergraduate](https://github.com/CTF-Archives/2024-YCB-Undergraduate) 
>
> 参赛队伍：Del0n1x；ranking：23/500（竞争太激烈力，打麻了😭）
>
> 更新：最后居然第十七名进决赛了wow😭，真是太不容易了
>
> 队伍[writeup](https://www.yuque.com/keyboard-ovrmx/scxvuu/rckx59lv20a3h6uy?singleDoc#)已公开访问，欢迎师傅们参考交流

## NLP_Model_Attack

> 题目分值：已答出45次，初始分值500.0，当前分值483.2，解出分值482.43 题目难度：中等

预训练三分类模型distilBert，攻击策略为经典PWWSRen2019

使用[TextAttack](https://github.com/QData/TextAttack)工具箱，详细教程和代码示例参考[TextAttack Documentation — TextAttack 0.3.10 documentation](https://textattack.readthedocs.io/en/master/)，各类API文档使用方法写得很好，writeup也可以写得很简洁。

## Targeted_Image_adv_attacks

> 题目分值：已答出10次，初始分值500.0，当前分值499.3，解出分值499.13 题目难度：困难

预训练模型为densenet121_catdogfox_classify，攻击策略为PGD，各项扰动参数为eps=4/255, alpha=0.5/255, steps=12, random_start=True

使用[torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch)工具箱，好用爱用，

重写数据加载方法，PGD CW策略参数调试，150张图片改写批量运行，三类图片分别攻击，运气也很好PGD策略能本地过测，最后提交多一点点过关。

结果通过Nvidia RTX2080TI 本地计算Average SSIM: 0.9547 Average Attack Success Rate: 0.9200

经过优化能在17s左右生成150张对抗样本且满足上述数据效果，代码是经过claude-3.5-sonnet简化的版本，环境参考[AdvAttack requirements](https://github.com/Harry24k/adversarial-attacks-pytorch)。

