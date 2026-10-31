#!/usr/bin/env python3
"""
AIS模块硬件检测工具 - 完整版
"""

import serial
import time
import os
import subprocess

class AISHardwareDetector:
    def __init__(self):
        self.ports = ['/dev/ttyS1', '/dev/ttyS2', '/dev/ttyS4', '/dev/ttyS8']
        self.results = {}
        
    def check_port_exists(self, port):
        """检查端口是否存在"""
        return os.path.exists(port)
    
    def check_port_permissions(self, port):
        """检查端口权限"""
        try:
            return os.access(port, os.R_OK | os.W_OK)
        except:
            return False
    
    def check_port_usage(self, port):
        """检查端口是否被占用"""
        try:
            result = subprocess.run(
                ['sudo', 'fuser', port],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                return True, result.stdout.strip()
            return False, None
        except:
            return False, None
    
    def test_ais_module(self, port, baudrate=115200, timeout=3):
        """测试AIS模块"""
        try:
            ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout
            )
            
            # 清空缓冲区
            ser.reset_input_buffer()
            
            # 发送配置指令 (可选)
            # 配置为A频道接收
            # ser.write(b"$PGNS,M,A*06\r\n")
            
            # 等待响应
            start_time = time.time()
            data_received = []
            
            while time.time() - start_time < timeout:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('ascii', errors='ignore').strip()
                    if line:
                        data_received.append(line)
                        if line.startswith(('!', '$')):
                            print(f"✅ {port}: 收到AIS数据!")
                            print(f"   {line[:80]}")
            
            ser.close()
            
            return {
                'success': len(data_received) > 0,
                'data': data_received,
                'count': len(data_received)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': []
            }
    
    def run_diagnostic(self):
        """运行完整诊断"""
        print("=" * 60)
        print("🔧 AIS模块硬件诊断工具")
        print("=" * 60)
        
        for port in self.ports:
            print(f"\n📌 检查 {port}")
            print("-" * 40)
            
            # 1. 检查端口是否存在
            exists = self.check_port_exists(port)
            print(f"  存在: {'✅' if exists else '❌'}")
            
            if not exists:
                continue
            
            # 2. 检查权限
            permissions = self.check_port_permissions(port)
            print(f"  权限: {'✅' if permissions else '❌'}")
            
            if not permissions:
                print("  建议: sudo chmod 666 " + port)
            
            # 3. 检查是否被占用
            used, pid = self.check_port_usage(port)
            if used:
                print(f"  ⚠️  被占用: PID {pid}")
            else:
                print(f"  ✅ 空闲")
            
            # 4. 测试串口硬件（回环测试需要手动短接）
            # 这里只做读取测试
            print(f"  测试AIS模块...")
            result = self.test_ais_module(port)
            
            if result['success']:
                print(f"  ✅ 检测到数据 ({result['count']} 条)")
                self.results[port] = result
            else:
                if 'error' in result:
                    print(f"  ❌ 错误: {result['error']}")
                else:
                    print(f"  ❌ 无数据 (可能没有AIS信号或连接问题)")
        
        # 总结
        self.print_summary()
    
    def print_summary(self):
        """打印诊断总结"""
        print("\n" + "=" * 60)
        print("📊 诊断总结")
        print("=" * 60)
        
        found = False
        for port, result in self.results.items():
            if result.get('success', False):
                print(f"✅ {port}: 正常工作")
                print(f"   收到 {result['count']} 条数据")
                if result['data']:
                    print(f"   示例: {result['data'][0][:60]}")
                found = True
        
        if not found:
            print("\n⚠️ 未检测到AIS数据，可能的原因:")
            print("  1. AIS模块未供电 (检查12V)")
            print("  2. 接线错误 (RX-TX交叉)")
            print("  3. 引脚7未接地 (处于发射模式)")
            print("  4. 没有AIS信号 (检查天线)")
            print("  5. 波特率不匹配 (应为115200)")
            print("  6. 模块故障")
            print("\n建议操作:")
            print("  1. 检查硬件连接")
            print("  2. 使用示波器测量引脚3 (TX) 是否有数据")
            print("  3. 尝试其他串口")
            print("  4. 检查AIS模块指示灯状态")

if __name__ == "__main__":
    detector = AISHardwareDetector()
    detector.run_diagnostic()