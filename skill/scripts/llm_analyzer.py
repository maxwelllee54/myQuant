#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM增强分析模块
支持API和浏览器自动化两种模式
"""

import os
import sys
from typing import List, Dict, Optional

class LLMAnalyzer:
    """大语言模型分析器"""
    
    def __init__(self, api_mode='auto'):
        """
        初始化LLM分析器
        
        Args:
            api_mode: 'api', 'browser', 'auto'
                - 'api': 仅使用API
                - 'browser': 仅使用浏览器
                - 'auto': 自动选择（优先API，失败时切换浏览器）
        """
        self.api_mode = api_mode
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
    
    def analyze_stock_gemini_api(self, ticker: str, company_name: str, prompt: str) -> Optional[str]:
        """使用Gemini API分析股票"""
        if not self.gemini_api_key:
            print("⚠️  未找到GEMINI_API_KEY环境变量")
            return None
        
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=self.gemini_api_key)
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=4096
                )
            )
            
            return response.text
            
        except Exception as e:
            print(f"✗ Gemini API调用失败: {e}")
            return None
    
    def analyze_stock_openai_api(self, ticker: str, company_name: str, prompt: str) -> Optional[str]:
        """使用OpenAI API分析股票"""
        if not self.openai_api_key:
            print("⚠️  未找到OPENAI_API_KEY环境变量")
            return None
        
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一位资深的投资分析师。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"✗ OpenAI API调用失败: {e}")
            return None
    
    def analyze_stock_browser(self, ticker: str, company_name: str, prompt: str, 
                            service: str = 'chatgpt') -> Optional[str]:
        """
        使用浏览器自动化分析股票
        
        Args:
            service: 'chatgpt', 'claude', 'gemini'
        """
        print(f"⚠️  API模式失败，切换到浏览器自动化模式")
        print(f"📌 请确保您已在浏览器中登录 {service}")
        print(f"📌 如需帮助，可以手动接管浏览器完成分析")
        
        # 这里应该调用浏览器自动化工具
        # 由于这是框架代码，实际实现需要在调用时完成
        
        return None
    
    def analyze_stock(self, ticker: str, company_name: str, prompt: str, 
                     models: List[str] = ['gemini', 'openai']) -> Dict[str, str]:
        """
        使用多个LLM分析股票
        
        Args:
            ticker: 股票代码
            company_name: 公司名称
            prompt: 分析提示词
            models: 要使用的模型列表
        
        Returns:
            Dict[model_name, analysis_result]
        """
        results = {}
        
        for model in models:
            print(f"\n正在使用 {model.upper()} 分析 {ticker} ({company_name})...")
            
            if model == 'gemini':
                if self.api_mode in ['api', 'auto']:
                    result = self.analyze_stock_gemini_api(ticker, company_name, prompt)
                    if result:
                        results['gemini'] = result
                        print(f"✓ Gemini分析完成")
                        continue
                
                if self.api_mode in ['browser', 'auto']:
                    result = self.analyze_stock_browser(ticker, company_name, prompt, 'gemini')
                    if result:
                        results['gemini_browser'] = result
            
            elif model == 'openai':
                if self.api_mode in ['api', 'auto']:
                    result = self.analyze_stock_openai_api(ticker, company_name, prompt)
                    if result:
                        results['openai'] = result
                        print(f"✓ OpenAI分析完成")
                        continue
                
                if self.api_mode in ['browser', 'auto']:
                    result = self.analyze_stock_browser(ticker, company_name, prompt, 'chatgpt')
                    if result:
                        results['openai_browser'] = result
        
        return results

def main():
    """示例用法"""
    analyzer = LLMAnalyzer(api_mode='auto')
    
    prompt = """
    请分析以下股票的投资价值：
    
    股票代码: AAPL
    公司名称: Apple Inc.
    
    请从基本面、估值、增长前景和风险四个维度进行分析。
    """
    
    results = analyzer.analyze_stock('AAPL', 'Apple Inc.', prompt, models=['gemini'])
    
    for model, analysis in results.items():
        print(f"\n{'='*80}")
        print(f"{model.upper()} 分析结果:")
        print(f"{'='*80}")
        print(analysis)

if __name__ == '__main__':
    main()
