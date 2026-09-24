使用说明：


1.安装python3.9以上环境，3.13,3.14更好。

2.使用前建议先配置python国内源：cmd中输入：pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

3.完成上述工作后，在需要检测的目录下，用cmd输入:python check_topology(jiangsu).py "需检查矢量路径" -o "保存报告路径"

用途：
①可用于矢量自相交、重叠、有属性无图斑（空几何）等问题检查（重点检查前面三类，其他的可作为参考）。
②建议分县批量审核，速度快。
③代码递归检测矢量，可在上一级目录运行，实现批量化检测、可多开。
④自动形成矢量图斑问题报告（csv、txt双格式）。


--jsdczdrzy
