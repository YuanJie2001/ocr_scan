"""
身份证字段解析模块

从 OCR 识别结果中提取结构化的身份证信息字段。
"""
import logging
import re
from dataclasses import dataclass, field

from ocr_engine import OCRResult

logger = logging.getLogger(__name__)


@dataclass
class IDCardInfo:
    """身份证信息数据类"""
    name: str = ""              # 姓名
    gender: str = ""            # 性别
    ethnicity: str = ""         # 民族
    birth_date: str = ""        # 出生日期
    address: str = ""           # 住址
    id_number: str = ""         # 身份证号码
    raw_texts: list[str] = field(default_factory=list)  # 原始 OCR 文本

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "姓名": self.name or "未识别",
            "性别": self.gender or "未识别",
            "民族": self.ethnicity or "未识别",
            "出生": self.birth_date or "未识别",
            "住址": self.address or "未识别",
            "公民身份号码": self.id_number or "未识别",
        }

    def is_valid(self) -> bool:
        """检查是否至少识别出了身份证号码"""
        return bool(self.id_number)


class IDCardParser:
    """
    身份证字段解析器

    使用关键字匹配 + 正则表达式从 PaddleOCR 的结果中提取结构化字段。
    """

    # 身份证号码正则 (18位)
    _ID_PATTERN = re.compile(r"\d{17}[\dXx]")

    # 出生日期正则
    _BIRTH_PATTERN = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")

    # 性别关键字
    _GENDERS = {"男", "女"}

    # 民族列表（简化版）
    _ETHNICITIES = {
        "汉", "蒙古", "回", "藏", "维吾尔", "苗", "彝", "壮", "布依", "朝鲜",
        "满", "侗", "瑶", "白", "土家", "哈尼", "哈萨克", "傣", "黎", "傈僳",
        "佤", "畲", "高山", "拉祜", "水", "东乡", "纳西", "景颇", "柯尔克孜",
        "土", "达斡尔", "仫佬", "羌", "布朗", "撒拉", "毛南", "仡佬", "锡伯",
        "阿昌", "普米", "塔吉克", "怒", "乌孜别克", "俄罗斯", "鄂温克", "德昂",
        "保安", "裕固", "京", "塔塔尔", "独龙", "鄂伦春", "赫哲", "门巴",
        "珞巴", "基诺",
    }

    # 关键字 → 字段映射
    _FIELD_KEYWORDS = {
        "姓名": "name",
        "性别": "gender",
        "民族": "ethnicity",
        "出生": "birth_date",
        "住址": "address",
        "公民身份号码": "id_number",
    }

    def parse(self, ocr_results: list[OCRResult]) -> IDCardInfo:
        """
        从 OCR 结果中解析身份证信息

        Args:
            ocr_results: OCR 识别结果列表

        Returns:
            解析出的身份证信息
        """
        info = IDCardInfo()
        info.raw_texts = [r.text for r in ocr_results]

        # 按 Y 坐标排序（从上到下）
        sorted_results = sorted(ocr_results, key=lambda r: self._get_y(r))

        # 第一遍：通过关键字匹配提取字段
        address_parts = []
        for i, result in enumerate(sorted_results):
            text = result.text.strip()

            # 尝试匹配关键字
            for keyword, field_name in self._FIELD_KEYWORDS.items():
                if keyword in text:
                    value = text.split(keyword, 1)[-1].strip()
                    # 去除可能的冒号
                    value = value.lstrip(":：").strip()

                    if field_name == "name" and not info.name and value:
                        info.name = value
                    elif field_name == "gender" and not info.gender:
                        # 性别和民族可能在同一行
                        info.gender, info.ethnicity = self._parse_gender_ethnicity(value, text)
                    elif field_name == "ethnicity" and not info.ethnicity and value:
                        info.ethnicity = self._clean_ethnicity(value)
                    elif field_name == "birth_date" and not info.birth_date:
                        info.birth_date = self._parse_birth_date(value, text)
                    elif field_name == "address" and value:
                        address_parts.append(value)
                    elif field_name == "id_number":
                        id_num = self._extract_id_number(value) or self._extract_id_number(text)
                        if id_num:
                            info.id_number = id_num

            # 独立匹配身份证号码（可能单独成行）
            if not info.id_number:
                id_num = self._extract_id_number(text)
                if id_num:
                    info.id_number = id_num

        # 地址可能跨多行
        if address_parts:
            info.address = "".join(address_parts)

        # 第二遍：补充未匹配的字段
        self._fill_missing_fields(info, sorted_results)

        logger.info("身份证信息解析完成: 姓名=%s, 号码=%s", info.name, info.id_number)
        return info

    def _get_y(self, result: OCRResult) -> float:
        """获取识别结果的 Y 坐标（取左上角）"""
        if result.position and len(result.position) >= 1:
            return result.position[0][1]
        return 0.0

    def _parse_gender_ethnicity(self, value: str, full_text: str) -> tuple[str, str]:
        """
        解析性别和民族（它们通常在同一行）

        Returns:
            (性别, 民族)
        """
        gender = ""
        ethnicity = ""

        # 在完整文本中搜索
        search_text = full_text if not value else value

        for g in self._GENDERS:
            if g in search_text:
                gender = g
                break

        # 尝试在 "民族" 关键字后提取
        if "民族" in search_text:
            eth_part = search_text.split("民族", 1)[-1].strip().lstrip(":：").strip()
            ethnicity = self._clean_ethnicity(eth_part)
        else:
            # 尝试直接匹配民族名
            for eth in self._ETHNICITIES:
                if eth in search_text and eth != gender:
                    ethnicity = eth
                    break

        return gender, ethnicity

    def _clean_ethnicity(self, value: str) -> str:
        """清理民族字段"""
        # 去除 "族" 字和多余内容
        value = value.replace("族", "").strip()
        # 匹配已知民族
        for eth in self._ETHNICITIES:
            if eth in value:
                return eth
        return value[:4] if value else ""  # 最多取前4个字

    def _parse_birth_date(self, value: str, full_text: str) -> str:
        """解析出生日期"""
        search_text = full_text if not value else value
        match = self._BIRTH_PATTERN.search(search_text)
        if match:
            year, month, day = match.groups()
            return f"{year}年{int(month)}月{int(day)}日"
        # fallback：直接用数字组合
        digits = re.findall(r"\d+", search_text)
        if len(digits) >= 3:
            return f"{digits[0]}年{digits[1]}月{digits[2]}日"
        return value

    def _extract_id_number(self, text: str) -> str:
        """从文本中提取身份证号码"""
        # 清除空格和常见 OCR 误识别字符
        cleaned = text.replace(" ", "").replace("·", "").replace("-", "")
        match = self._ID_PATTERN.search(cleaned)
        if match:
            return match.group(0).upper()  # X 统一大写
        return ""

    def _fill_missing_fields(self, info: IDCardInfo, sorted_results: list[OCRResult]) -> None:
        """
        补全未通过关键字匹配到的字段

        当某些字段标签和值分成了不同的 OCR 行时使用。
        """
        for i, result in enumerate(sorted_results):
            text = result.text.strip()
            next_text = sorted_results[i + 1].text.strip() if i + 1 < len(sorted_results) else ""

            if text in self._FIELD_KEYWORDS and next_text:
                field_name = self._FIELD_KEYWORDS[text]

                if field_name == "name" and not info.name:
                    if not any(k in next_text for k in self._FIELD_KEYWORDS):
                        info.name = next_text
                elif field_name == "gender":
                    if not info.gender or not info.ethnicity:
                        gender, ethnicity = self._parse_gender_ethnicity(next_text, next_text)
                        if gender and not info.gender:
                            info.gender = gender
                        if ethnicity and not info.ethnicity:
                            info.ethnicity = ethnicity
                elif field_name == "ethnicity" and not info.ethnicity:
                    if not any(k in next_text for k in self._FIELD_KEYWORDS):
                        info.ethnicity = self._clean_ethnicity(next_text)
                elif field_name == "birth_date" and not info.birth_date:
                    if not any(k in next_text for k in self._FIELD_KEYWORDS):
                        info.birth_date = self._parse_birth_date(next_text, next_text)
                elif field_name == "address" and not info.address:
                    parts = []
                    j = i + 1
                    while j < len(sorted_results):
                        candidate = sorted_results[j].text.strip()
                        if not candidate:
                            j += 1
                            continue
                        if any(k in candidate for k in self._FIELD_KEYWORDS):
                            break
                        if self._extract_id_number(candidate):
                            break
                        parts.append(candidate)
                        j += 1
                    if parts:
                        info.address = "".join(parts)
                elif field_name == "id_number" and not info.id_number:
                    id_num = self._extract_id_number(next_text)
                    if id_num:
                        info.id_number = id_num
