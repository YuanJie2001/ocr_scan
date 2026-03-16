"""
身份证号码校验模块

提供身份证号码合法性校验和信息提取功能。
"""
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class IDCardValidator:
    """身份证号码校验器"""

    # 省份编码表
    PROVINCE_CODES = {
        "11": "北京", "12": "天津", "13": "河北", "14": "山西", "15": "内蒙古",
        "21": "辽宁", "22": "吉林", "23": "黑龙江",
        "31": "上海", "32": "江苏", "33": "浙江", "34": "安徽", "35": "福建",
        "36": "江西", "37": "山东",
        "41": "河南", "42": "湖北", "43": "湖南", "44": "广东", "45": "广西",
        "46": "海南",
        "50": "重庆", "51": "四川", "52": "贵州", "53": "云南", "54": "西藏",
        "61": "陕西", "62": "甘肃", "63": "青海", "64": "宁夏", "65": "新疆",
        "71": "台湾", "81": "香港", "82": "澳门",
    }

    # 校验码权重
    _WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    _CHECK_CODES = "10X98765432"

    @classmethod
    def validate(cls, id_number: str) -> tuple[bool, str]:
        """
        校验身份证号码合法性

        Args:
            id_number: 身份证号码字符串

        Returns:
            (是否合法, 错误信息)
        """
        id_number = id_number.strip().upper()

        # 1. 格式校验
        if not re.match(r"^\d{17}[\dX]$", id_number):
            return False, "格式不正确（应为18位，末位可为数字或X）"

        # 2. 省份校验
        province_code = id_number[:2]
        if province_code not in cls.PROVINCE_CODES:
            return False, f"省份代码不正确: {province_code}"

        # 3. 出生日期校验
        try:
            year = int(id_number[6:10])
            month = int(id_number[10:12])
            day = int(id_number[12:14])
            birth_date = datetime(year, month, day)
            if birth_date > datetime.now():
                return False, "出生日期超过当前日期"
        except ValueError:
            return False, f"出生日期不合法: {id_number[6:14]}"

        # 4. 校验码校验
        weighted_sum = sum(int(id_number[i]) * cls._WEIGHTS[i] for i in range(17))
        expected_check = cls._CHECK_CODES[weighted_sum % 11]

        if expected_check != id_number[17]:
            return False, f"校验码错误（期望 {expected_check}，实际 {id_number[17]}）"

        return True, ""

    @classmethod
    def extract_info(cls, id_number: str) -> dict:
        """
        从身份证号码中提取基本信息

        Args:
            id_number: 已校验合法的身份证号码

        Returns:
            包含省份、出生日期、性别等信息的字典
        """
        id_number = id_number.strip().upper()

        is_valid, error_msg = cls.validate(id_number)
        if not is_valid:
            logger.warning("身份证号码校验失败: %s", error_msg)
            return {"error": error_msg}

        province = cls.PROVINCE_CODES.get(id_number[:2], "未知")
        birth_date = f"{id_number[6:10]}-{id_number[10:12]}-{id_number[12:14]}"
        gender = "男" if int(id_number[16]) % 2 == 1 else "女"

        return {
            "province": province,
            "birth_date": birth_date,
            "gender": gender,
            "province_code": id_number[:2],
            "city_code": id_number[2:4],
            "district_code": id_number[4:6],
            "sequence": id_number[14:17],
            "check_code": id_number[17],
        }
