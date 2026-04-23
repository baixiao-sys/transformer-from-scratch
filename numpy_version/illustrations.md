### 说明
1. 本numpy版本为transformer基础原始版，只实现了***SPE***，***RoPE***，***MHA***,***FFN***,***LayerNorm***,***Linear***各基础模块的numpy版本，暂且只搭建了DecoderOnly的架构。
2. 此版本仅作为深入了解数学原理学习使用，更多功能扩展和调试在torch版本中实现。
3. 本版本未作GPU迁移以及相关提速优化，因此不建议使用本版本训练。