/*!
 * chartjs-adapter-luxon v0.2.1
 * 简化版, 专为交易系统定制
 * 修复日期适配器错误的版本
 */
(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(require('chart.js'), require('luxon')) :
    typeof define === 'function' && define.amd ? define(['chart.js', 'luxon'], factory) :
    (global = global || self, factory(global.Chart, global.luxon));
}(this, (function (Chart, luxon) { 'use strict';

    Chart = Chart && Chart.hasOwnProperty('default') ? Chart['default'] : Chart;
    luxon = luxon && Object.prototype.hasOwnProperty.call(luxon, 'default') ? luxon['default'] : luxon;

    console.log("加载Chart.js Luxon适配器 v0.2.1");
    
    // 确保Luxon和Chart已加载
    if (typeof luxon === 'undefined') {
        console.error("Luxon库未加载！");
        return;
    }
    
    if (typeof Chart === 'undefined') {
        console.error("Chart.js库未加载！");
        return;
    }
    
    const DateTime = luxon.DateTime;
    const Duration = luxon.Duration;

    const FORMATS = {
        datetime: 'yyyy-MM-dd HH:mm:ss',
        millisecond: 'HH:mm:ss.SSS',
        second: 'HH:mm:ss',
        minute: 'HH:mm',
        hour: 'HH:mm',
        day: 'yyyy-MM-dd',
        week: 'yyyy-MM-dd',
        month: 'yyyy-MM',
        quarter: 'yyyy-qq',
        year: 'yyyy'
    };

    // 确保_adapters对象存在
    if (!Chart._adapters) {
        Chart._adapters = {};
    }
    
    const adapter = {
        _id: 'luxon', // 适配器标识符
        
        // 解析日期
        parse: function(value, format) {
            if (value === null || value === undefined) {
                return null;
            }
            
            if (value instanceof DateTime) {
                return value;
            }
            
            if (typeof value === 'string' && typeof format === 'string') {
                return DateTime.fromFormat(value, format);
            }
            
            if (value instanceof Date) {
                return DateTime.fromJSDate(value);
            }
            
            if (typeof value === 'number') {
                return DateTime.fromMillis(value);
            }
            
            return DateTime.fromISO(value);
        },
        
        // 格式化日期
        format: function(time, format) {
            if (!time || !time.isValid) {
                return '';
            }
            return time.toFormat(format || FORMATS.datetime);
        },
        
        // 添加时间
        add: function(time, amount, unit) {
            if (!time || !time.isValid) {
                return null;
            }
            const args = {};
            args[unit] = amount;
            return time.plus(args);
        },
        
        // 获取时间段开始
        startOf: function(time, unit, weekday) {
            if (!time || !time.isValid) {
                return null;
            }
            
            if (unit === 'isoWeek') {
                // ISO周的开始是周一
                return time.startOf('week').set({weekday: weekday || 1});
            }
            return time.startOf(unit);
        },
        
        // 获取时间段结束
        endOf: function(time, unit) {
            if (!time || !time.isValid) {
                return null;
            }
            return time.endOf(unit);
        },
        
        // 计算时间差
        diff: function(max, min, unit) {
            if (!max || !max.isValid || !min || !min.isValid) {
                return null;
            }
            return max.diff(min).as(unit);
        },
        
        // 创建日期
        _create: function(time) {
            if (time === undefined || time === null) {
                return DateTime.local();
            }
            return this.parse(time);
        },
        
        // 获取时间戳
        valueOf: function(time) {
            if (!time || !time.isValid) {
                return null;
            }
            return time.toMillis();
        }
    };

    // 正确设置适配器
    if (Chart._adapters) {
        // 直接设置 _adapters.date
        Chart._adapters._date = adapter;
    } else if (Chart.registry && Chart.registry._adapters) {
        // Chart.js v3+ 的注册方式
        Chart.registry._adapters.date = adapter;
    } else {
        // 如果以上都失败，尝试直接在Chart对象上创建_adapters
        Chart._adapters = { _date: adapter };
    }
    
    // 设置默认适配器
    if (Chart.defaults && Chart.defaults.scales && Chart.defaults.scales.time) {
        Chart.defaults.scales.time.adapters = Chart.defaults.scales.time.adapters || {};
        Chart.defaults.scales.time.adapters.date = adapter;
    }
    
    console.log("Chart.js Luxon适配器初始化完成");

})));
