# roaring32

一个 32 位无符号整数集合库，采用 Roaring Bitmap 风格的高 16 位分片设计：
每个分片按基数与密度在 **array / bitmap / run** 三种容器之间自动转换。

- 零依赖，TypeScript + Node.js，不引入任何 bitmap 第三方库（位图直接用 `Uint32Array`）。
- 公开增删、包含、秩（rank）、选择（select）、范围迭代，并集 / 交集 / 差集 / 对称差（函数式与原地两套）。
- 集合运算按两侧容器类型组合分派专用算法（有序归并、按字位运算、区间代数），不盲目展开成数组。
- 版本化二进制序列化，解码前先校验魔数、版本、偏移、长度与基数。
- 范围迭代使用可回收的临时缓冲，调用方提前取消（`break` / `return()`）时通过 `finally` 释放。

## 构建与测试

```bash
npm install
npm run build   # tsc -> dist/
npm test        # node --test dist/test/
```

## 快速开始

```ts
import { RoaringSet } from './src/index';

const a = RoaringSet.of(0, 1, 2, 65536, 0xffffffff);
a.addRange(1_000_000, 2_000_000);

a.has(65536);        // true
a.rank(2_000_000);   // 元素个数 <= x
a.select(0);         // 第 0 个（最小）元素

for (const v of a.iterateRange(1_500_000, 1_600_000)) {
  // ...
}

const b = RoaringSet.from(/* 任意可迭代对象 */);
a.union(b);                       // 非原地，返回新集合
a.unionInPlace(b);                // 原地版本（四种运算均提供）

const bytes = a.serialize();      // Uint8Array 版本化二进制
const c = RoaringSet.deserialize(bytes);
```

## 设计要点

### 分片与容器

| 容器 | 表示 | 适用密度 |
| --- | --- | --- |
| `ArrayContainer` | 有序 `Uint16Array`，容量翻倍增长 | 基数 ≤ 4096 |
| `BitmapContainer` | 2048 个 u32 字（65536 位） | 基数 > 4096 |
| `RunContainer` | 有序、不重叠、不相邻的 `(start, length)` | 连续区间占主导时 |

- 点添加/删除跨越阈值（4096）时自动 array ↔ bitmap 转换；转换例程保持值有序、基数不变。
- 集合运算结果经 `normalize()` 按序列化字节数选择最紧凑的表示（平局保留原类型）；`runOptimize()` 可主动压缩。

### 集合运算分派

`src/ops.ts` 中每种运算（and / or / andNot / xor）对 3×3 容器组合各有专用实现，例如：

- array × array：双指针有序归并；
- bitmap × bitmap：逐字 `& / | / ^ / &~`，`popcount` 直接得基数；
- run × run：区间交 / 并（合并相邻区间）/ 差；
- run × bitmap：对区间做位掩码读写，不展开 run。

原地变体（`containerIOr` 等）复用左侧存储：bitmap 直接逐字修改、array 反向归并并就地压实、run 替换区间表；不可原地高效完成的组合回退到函数式实现。原地运算从不修改右侧操作数。

### 二进制格式（版本 1）

```
偏移   长度    字段
0      4       魔数 "RB32"（小端 0x32334252）
4      4       格式版本号
8      4       容器数量 N
12     4       flags（保留，必须为 0）
16     2N      分片键（u16，严格递增）
+2N    2N      基数 - 1（u16）
+4N    N       容器类型（u8：0=array, 1=bitmap, 2=run）
       （对齐到 4 字节）
...    4N      各容器负载绝对偏移（u32，严格递增、连续）
...    ...     负载：
                 array : card × u16 严格递增值
                 bitmap: 2048 × u32 位图字
                 run   : runs × (u16 start, u16 length)
```

解码顺序：魔数 / 版本 / flags → 键严格递增 → 偏移连续且在缓冲范围内 → 每段长度与类型、声明基数匹配 → 最后才解析负载内容（array 严格递增、bitmap popcount 等于声明值、run 有序不重叠且基数吻合）。任何一步失败抛 `DeserializationError`，不产生半成品集合。

### 范围迭代与缓冲释放

`iterateRange(start, end)` 对覆盖到的 bitmap 分片从全局 `wordPool` 借出一块 u32 scratch，先复制要扫描的字，使迭代与潜在修改隔离；生成器在正常结束、`break` 提前退出或显式 `iterator.return()` 时都会经 `finally` 归还缓冲。可通过 `wordPool.totalAcquired / totalReleased` 观测。

## 测试覆盖（`test/roaring.test.ts`）

node:test，16 个用例，含：

- 阈值来回转换（4095/4096/4097 多轮往返，顺序与基数严格校验）；
- 0、2^32−1、`addRange(0, 2^32−1)` 全空间边界；
- 跨大量分片的长连续区间（run 压缩 + 序列化往返）；
- 稀疏交集（两个 5k 随机集合）与完全不相交集合；
- 八种随机场景下四种运算 × 函数式/原地，全部对拍朴素 JS Set，并校验右操作数不变；
- rank/select 对拍朴素集合：稀疏、稠密、run、边界多密度随机探测；
- 4000 步随机操作序列（add/remove/addRange/has/rank/select/runOptimize）对拍 JS Set；
- 13 类损坏输入（魔数、版本、flags、计数、键序、基数、类型、越界/错位偏移、负载乱序、popcount 不符、run 重叠、截断、空缓冲）；
- 百万元素：顺序点插、1M 随机去重、大集合代数与序列化；
- 迭代器 `break` / `return()` / 正常结束三种路径的缓冲释放计数。
