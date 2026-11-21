# 案例分析（中文）

分析对象是当前正式 run 中由 `case_seed=2026` 固定抽取的 10 篇文档。下文的“命中”按项目实际评价规则计算：小写、正则分词、Porter stemming 后作精确短语匹配。TF-IDF 是本组基线；TextRank 窗口 2 是主设置，窗口 5 是单因素窗口对照。窗口 3 和 10 的完整预测仍见同目录的 `cases_10.jsonl`。

每例的输入摘录都很短；标签按当前匹配规则分成“摘要中可匹配”和“摘要中未匹配”。未匹配标签不能由只抽取摘要原文短语的方法直接命中。预测列展示各方法前 10 名；案例文件保留完整排序。

## 案例 1：ID 1960——行人疏散与幂律行为

**输入摘录：** “We analyze the exit dynamics of pedestrians who are initially confined in a room.”

**参考关键词（摘要中可匹配）：** streaming; cellular automata; iteration time; adjacent cell vacancy; arching signatures; jamming; burst-size frequency distribution; collective slow-down; disruptive interference; power-law behavior; exit dynamics; confined pedestrians.

**参考关键词（摘要中未匹配）：** self-organised criticality.

**TF-IDF@10：** w; pedestrians; exit; q; room; confined pedestrians; exit dynamics; power-law; single-exit room; power-law behavior。命中：confined pedestrians、exit dynamics、power-law behavior。

**TextRank-2@10：** exit; room; cell; q; behavior; pedestrians; power-law behavior; sizes; capability; frequency。命中：power-law behavior。

**TextRank-5@10：** exit; w; pedestrians; q; average exit; power-law; power-law behavior; behavior; cell; room。命中：power-law behavior。

**错误类别与解释：** 泛化词和数学符号噪声。算法抓到了“power-law behavior”，但把 `q`、`w`、room、cell 等排得很高。摘要讨论拥堵 burst 的关系和参数，单词共现图容易把局部高频词提上来；符号也被当前分词与词性处理保留下来。窗口 2 和 5 在此例都只命中 1 个关键词。

## 案例 2：ID 2——Wavelength services

**输入摘录：** “Wavelength services have been hyped ad nauseam for years.”

**参考关键词（摘要中可匹配）：** wavelength services.

**参考关键词（摘要中未匹配）：** fiber optic networks; looking glass networks; pointeast research.

**TF-IDF@10：** wavelength services; services; industry; nauseam; turn-up; lukewarm reception; flourish。命中：wavelength services。

**TextRank-2@10：** turn-up; industry; services; margins; such services; ad nauseam; lukewarm。命中：无。

**TextRank-5@10：** services; industry; reasons; impressive margins; time; many; lukewarm。命中：无。

**错误类别与解释：** 稀有实体遗漏，以及参考标签超出摘要可抽取范围。三个参考词组在摘要文本中无法直接匹配；可抽取的 `wavelength services` 被 TF-IDF 排第一，但 TextRank 窗口 2 和 5 更偏向 services、industry 等常见词。窗口 10 把 wavelength services 提至第 2 名，说明扩大的共现范围在本例有帮助，但仍无法恢复摘要里没有的三个标签。

## 案例 3：ID 2016——相机标定中的 ROI 定位

**输入摘录：** “We present an automatic method for region of interest (ROI) location in camera calibration.”

**参考关键词（摘要中可匹配）：** fully automatic algorithm; interest location; camera calibration; region of interest location; computer vision inspection; roi location algorithm; radon transform; calibration process; rotation angle; time/space consumption problem; fully automatic camera calibration; intrinsic parameters; zoom lens; camera parameters.

**参考关键词（摘要中未匹配）：** human interactive specification.

**TF-IDF@10：** calibration; camera calibration; automatic camera calibration; algorithm; roi; roi location algorithm; automatic algorithm; location; target; camera parameters。命中：camera calibration; roi location algorithm; camera parameters。

**TextRank-2@10：** roi; roi location algorithm; vision; target; rotation; improvement; automatic; automatic algorithm; automatic camera calibration; algorithm。命中：roi location algorithm。

**TextRank-5@10：** algorithm; roi location algorithm; roi; calibration; camera calibration; camera; target; location; method; interest location。命中：roi location algorithm; camera calibration; interest location。

**错误类别与解释：** 词组边界和过短候选。摘要中大多数标签可抽取，但 TextRank-2 把 ROI 和 calibration 的相关单词分散排序，没能把 `camera calibration` 提到前 10；窗口 5 排名更好。`algorithm`、`target` 等泛词也占据位置，挤掉了 `radon transform`、`rotation angle` 等较具体术语。

## 案例 4：ID 2069——企业控制结构

**输入摘录：** “Empirical research on the organization of firms requires that firms be classified on the basis of their control structures.”

**参考关键词（摘要中可匹配）：** ultimate control group; hierarchical organization; organizational structures; authority structures.

**参考关键词（摘要中未匹配）：** committees; control rights; firm organization.

**TF-IDF@10：** ultimate; ultimate control group; structures; control; organization; authority structures; firms; superiors; subordinates; mild。命中：ultimate control group; authority structures。

**TextRank-2@10：** structures; ultimate; organizational structures; economic; ultimate control group; ultimate control group empirical; such; choice procedures such; control; hierarchical。命中：organizational structures; ultimate control group。

**TextRank-5@10：** control; structures; ultimate; well-defined ultimate control group; ultimate control group empirical research; organization; such; hierarchical organization; author; hierarchy。命中：hierarchical organization。

**错误类别与解释：** 词组过度扩展和摘要外标签。TextRank-2 命中两个完整关键词，但也把关键词延长成 `ultimate control group empirical`；窗口 5 命中 `hierarchical organization`，同时产生 `well-defined ultimate control group` 这类与标签边界不一致的长短语。其余三个参考词组不在摘要中，不能期待抽取器仅凭摘要预测出来。

## 案例 5：ID 2190——服务发现标准

**输入摘录：** “Jini, Universal Plug and Play (UPnP), Salutation, and Service Location Protocol are among the front-runners.”

**参考关键词（摘要中可匹配）：** service discovery; jini; universal plug and play; salutation; service location protocol; pervasive computing.

**参考关键词（摘要中未匹配）：** mobile computing.

**TF-IDF@10：** service discovery; service; office; standards; home office; jini; upnp; salutation; front-runners; plug-and-play。命中：service discovery; jini; salutation。

**TextRank-2@10：** service discovery; office; service location; term service discovery; computing; small office; universal; s; jini; developers。命中：service discovery; jini。

**TextRank-5@10：** service; service discovery; office; user; mobile; home office; standards; jini; s potential; universal plug。命中：service discovery; jini。

**错误类别与解释：** 专名与缩写拆分。摘要里列举了多个标准名称，但 TextRank-2 只保留完整的 `jini`，把其他名称拆成 `universal`、`service location` 等不完整片段；`UPnP` 与完整标签 `universal plug and play` 也不是同一个精确短语。它同时偏好 office、computing 之类较常见节点。该例的参考关键词大多确实出现在摘要中，说明主要问题是排序和短语重建，而非标签不可抽取。

## 案例 6：ID 2167——软件专利立法

**输入摘录：** “The aim … is to harmonise national patent laws relating to inventions using software.”

**参考关键词（摘要中可匹配）：** european commission; directive on the protection by patents of computer-implemented inventions; national patent laws; eu; european patent office; national courts; computer programs.

**参考关键词（摘要中未匹配）：** law harmonisation.

**TF-IDF@10：** european; directive; european commission; european patent; draft directive; inventions; commission; european national courts; national patent; computer-implemented inventions。命中：european commission。

**TextRank-2@10：** european; european patent; european national; national patent; directive; computer-implemented inventions; inventions; extensive consultation; eu; computer。命中：eu。

**TextRank-5@10：** european; european national; european patent; directive; european commission; software; inventions; national patent; commission; protection。命中：european commission。

**错误类别与解释：** 重要长短语被切碎。模型注意到 european、patent、directive 等主题词，却未把 `european patent office` 或完整的超长 directive 标签排进前 10；窗口 2 只命中 `eu`，窗口 5 命中 `european commission`。`law harmonisation` 不在摘要中，其他词组的错失则体现了节点排序和短语边界不足。

## 案例 7：ID 277——偏序理论与 QSAR

**输入摘录：** “Partial order theory … allows ordering of compounds, based on selected structural and/or electronic descriptors.”

**参考关键词（摘要中可匹配）：** quantitative structure-activity relationships; partial order theory; linear extensions; electronic descriptors; modeled order; end points; solubilities.

**参考关键词（摘要中未匹配）：** predicting power improvement; structural descriptors; graphical representation; combinatorial rule; most probable linear order; partially ordered set; hasse diagram; organic compounds.

**TF-IDF@10：** compounds; order; partial order; noninvestigated compounds; experimental order; modeled order; linear extensions partial order theory; linear extensions; partial ordering; pot。命中：modeled order; linear extensions。

**TextRank-2@10：** order; partial order; experimental order; end-point; linear extensions partial order theory; partial; structure-activity; distribution; solubility; linear extensions。命中：solubility; linear extensions。

**TextRank-5@10：** order; partial order; experimental order; model order; compounds; linear extensions partial order theory; linear extensions; pot; partial ordering; theory。命中：model order; linear extensions。

**错误类别与解释：** 通用词占位与标签边界不一致。模型反复提取 `order`、`partial order` 和 compounds；这些词与主题有关，但在精确短语评估下不等于较长的 `partial order theory` 或其他标签。部分参考标签在摘要里没有直接出现，另有一些较具体标签未获高分。这个例子也提醒：语义相关不等于按本项目的精确词组指标命中。

## 案例 8：ID 357——Banach 空间上的模糊集嵌入

**输入摘录：** “We present an extension of the Minkowski embedding theorem … on a real separable Banach space X.”

**参考关键词（摘要中可匹配）：** isometric embedding; level-continuous fuzzy sets; real separable banach space; real continuous functions; cartesian product; unit ball; dual space; ascoli-arzela type theorem; cauchy problem.

**参考关键词（摘要中未匹配）：** compact-convex fuzzy sets.

**TF-IDF@10：** x; space x; banach space x; sub; t; banach; f; sub c; banach space; theorem。命中：无。

**TextRank-2@10：** x; sub; space x; fuzzy; type; banach; real separable banach space x; banach space; t; real continuous。命中：无。

**TextRank-5@10：** x; space x; banach space x; banach; sub; banach space; f; sub c; real; particular。命中：无。

**错误类别与解释：** 数学符号和格式噪声，是这 10 例中最明显的边界案例。摘要包含 `X*`、`F_sub c`、`[0,1]` 等公式结构；当前正则分词把 x、sub、f、t 等当成普通词节点，图排序于是被变量和格式碎片占据。此例所有方法在前 10 名都没有命中关键词。改进方向是处理数学记号或过滤单字符变量，并对数学领域文本单独评估。

## 案例 9：ID 395——同轴耦合器的保角映射设计

**输入摘录：** “Numerical conformal mapping is exploited as a simple, accurate, and efficient tool for the analysis and design of coaxial waveguides and couplers.”

**参考关键词（摘要中可匹配）：** coaxial couplers; complex cross section; coaxial waveguides; schwarz-christoffel toolbox; public-domain matlab package; slotted coaxial cables; symmetrical couplers; polygonal inner conductors; external shields; metallic diaphragms; nonsymmetrical couplers.

**参考关键词（摘要中未匹配）：** conformal mapping design tools; circular inner conductors; numerical conformal transformations.

**TF-IDF@10：** coaxial couplers; coaxial; couplers; coaxial cables; nonsymmetrical couplers; inner conductors; complex cross section; conformal-mapping; conformal; schwarz-christoffel。命中：coaxial couplers; nonsymmetrical couplers; complex cross section。

**TextRank-2@10：** coaxial; coaxial couplers; design; matlab; inner; additional; couplers; cross section numerical conformal; cross section; accurate。命中：coaxial couplers。

**TextRank-5@10：** couplers; coaxial couplers; coaxial; complex cross section; fraction; inner; design; finite-element method; excellent agreement; circular。命中：coaxial couplers; complex cross section。

**错误类别与解释：** 窗口敏感性和短语截断。窗口 2 只把 `coaxial couplers` 排入前 10；窗口 5 还命中 `complex cross section`；窗口 10 的完整预测又命中 `polygonal inner conductors` 与 `external shields`。更宽的共现范围在此文档确实找到更多特定术语，但仍产生 `cross section numerical conformal` 等过长短语。此个案支持“窗口可能影响排序”，不能单凭一个案例推断全体数据的趋势。

## 案例 10：ID 376——未知转移概率下的递归状态估计

**输入摘录：** “This work considers hybrid systems with continuous-valued target states and discrete-valued regime variable.”

**参考关键词（摘要中可匹配）：** recursive state estimation; multiple switching models; unknown transition probabilities; hybrid systems; continuous-valued target states; discrete-valued regime variable; finite state markov chain; random transition probabilities; dirichlet distributions; marginal posterior distribution; maneuvering target tracking.

**参考关键词（摘要中未匹配）：** 无；11 个标签都可在摘要中匹配。

**TF-IDF@10：** regime; transition probabilities; regime variable; unknown transition probabilities; state; recursive state estimation; recursive hybrid state estimation; hybrid; unknown; target。命中：unknown transition probabilities; recursive state estimation。

**TextRank-2@10：** transition; target; regime; state estimation; state markov; hybrid state estimation; switching; posterior; hybrid; changes。命中：无。

**TextRank-5@10：** state; states; hybrid; target states; transition probabilities; unknown transition probabilities; unknown; regime; target; hybrid state estimation。命中：unknown transition probabilities。

**错误类别与解释：** 领域短语被拆成高频节点。所有参考词都在摘要中，TF-IDF 找到两个完整词组；TextRank-2 的前 10 名主要是 transition、target、regime 等单词或不完整组合，未命中完整标签。窗口 5 恢复了 `unknown transition probabilities`，但 `recursive state estimation` 等词组仍没有进入前 10。这说明失败与标签不可抽取无关，更可能来自排序和短语组合。

## 总结

这些案例展示了几类不同原因：摘要外参考标签限制了可达召回；缩写、专名和数学符号会被切碎或误当成普通词；精确短语匹配会把语义相关但边界不同的输出计为未命中；窗口变化会影响特定案例，但方向并不一致。因而不能只用单个案例解释全体指标，也不能把未命中都归因于 TextRank 排序。完整数值、预测和逐文档标签仍以同目录机器生成的 `cases_10.jsonl` 与 `predictions.jsonl` 为准。
