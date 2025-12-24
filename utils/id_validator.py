# 身份证号码校验工具
import re
from datetime import datetime

class IDCardValidator:
    @staticmethod
    def validate_id_card(id_card):
        """
        校验身份证号码是否合法
        
        参数:
        id_card: 身份证号码字符串
        
        返回:
        (bool, str): (是否合法, 错误信息)
        """
        # 去除空格
        id_card = id_card.strip()
        
        # 基本格式检查
        if not re.match(r'^\d{17}[0-9Xx]$', id_card):
            return False, "身份证号码格式不正确（应为18位数字，最后一位可以是数字或X）"
        
        # 提取身份证信息
        province_code = id_card[:2]
        birth_year = id_card[6:10]
        birth_month = id_card[10:12]
        birth_day = id_card[12:14]
        
        # 检查省份代码 (简化版)
        valid_provinces = {
            '11': '北京', '12': '天津', '13': '河北', '14': '山西', '15': '内蒙古',
            '21': '辽宁', '22': '吉林', '23': '黑龙江',
            '31': '上海', '32': '江苏', '33': '浙江', '34': '安徽', '35': '福建', '36': '江西', '37': '山东',
            '41': '河南', '42': '湖北', '43': '湖南', '44': '广东', '45': '广西', '46': '海南',
            '50': '重庆', '51': '四川', '52': '贵州', '53': '云南', '54': '西藏',
            '61': '陕西', '62': '甘肃', '63': '青海', '64': '宁夏', '65': '新疆',
            '71': '台湾', '81': '香港', '82': '澳门'
        }
        
        if province_code not in valid_provinces:
            return False, f"省份代码不正确: {province_code}"
        
        # 检查出生日期
        try:
            birth_date = datetime(int(birth_year), int(birth_month), int(birth_day))
            # 检查日期是否合理（不超过当前日期）
            if birth_date > datetime.now():
                return False, "出生日期不能超过当前日期"
        except ValueError:
            return False, f"出生日期不正确: {birth_year}-{birth_month}-{birth_day}"
        
        # 校验码计算
        weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        check_codes = '10X98765432'
        
        # 计算加权和
        weighted_sum = sum(int(id_card[i]) * weights[i] for i in range(17))
        
        # 计算校验码
        calculated_check = check_codes[weighted_sum % 11]
        
        # 检查校验码
        if calculated_check != id_card[17].upper():
            return False, f"校验码错误，应为{calculated_check}，实际为{id_card[17]}"
        
        return True, ""


    @staticmethod
    def extract_info_from_id_card(id_card):
        """
        从身份证号码中提取信息
        
        参数:
        id_card: 身份证号码字符串
        
        返回:
        dict: 包含身份证信息的字典
        """
        # 先验证身份证号码
        is_valid, error_msg = validate_id_card(id_card)
        if not is_valid:
            return {"error": error_msg}
        
        # 提取信息
        info = {
            "province_code": id_card[:2],
            "city_code": id_card[2:4],
            "district_code": id_card[4:6],
            "birth_year": id_card[6:10],
            "birth_month": id_card[10:12],
            "birth_day": id_card[12:14],
            "birth_date": f"{id_card[6:10]}-{id_card[10:12]}-{id_card[12:14]}",
            "sequence": id_card[14:17],
            "gender": "男" if int(id_card[16]) % 2 == 1 else "女",
            "check_code": id_card[17]
        }
        
        return info