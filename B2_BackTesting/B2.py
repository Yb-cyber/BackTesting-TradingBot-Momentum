import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import data,add_indicators
import xgboost as xgb

def obtener_modelo_entrenado():
    model = xgb.XGBClassifier()
    model._estimator_type = "classifier"
    model.load_model("crypto_momentum_25tp.json")
    return model

def run_backtest(df, model, features, initial_capital=1000, leverage=5, fee_pct=0.0005, threshold=0.60):
    """
    df: DataFrame con columnas 'open', 'high', 'low', 'close' y fechas en el índice.
    model: Tu modelo entrenado (XGBoost).
    features: Las columnas que usa el modelo para predecir (mismo orden que en el entrenamiento).
    initial_capital: Dinero inicial en USDT.
    leverage: Apalancamiento (ej. 5x).
    fee_pct: Comisión del exchange (0.0005 = 0.05% estándar en Binance Taker).
    threshold: Nivel de confianza mínimo para entrar (ej. 0.60).
    """
    
    # 1. PREPARACIÓN
    capital = initial_capital
    equity_curve = [capital]
    trades = []
    in_position = False
    
    # Obtenemos todas las probabilidades de una vez para optimizar velocidad
    # (Asumimos que el modelo ya está entrenado)
    print("Generando predicciones para todo el periodo...")
    all_probs = model.predict_proba(df[features])[:, 1]
    
    # Parámetros de la estrategia (Tus reglas actuales)
    TP_PCT = 0.015  # 1.5%
    SL_PCT = 0.005  # 0.7%
    MAX_HOLD_CANDLES = 32 # Cierre forzado tras 12 horas (48 velas de 15m) si no toca nada
    
    # 2. BUCLE DE SIMULACIÓN
    # Iteramos vela por vela.
    # i es el índice actual.
    for i in range(len(df) - MAX_HOLD_CANDLES):
        
        # Si ya tengo capital <= 0, game over
        if capital <= 0:
            break
            
        current_prob = all_probs[i]
        current_close = df['close'].iloc[i]
        timestamp = df.index[i]
        
        # --- LÓGICA DE ENTRADA ---
        if not in_position and current_prob >= threshold:
            
            # Gestión de Posición: Usamos todo el capital disponible (Compuesto)
            # Ojo: En la vida real, quizás usarías solo una parte. Aquí simulamos "Full Margin".
            margin = capital 
            position_size = margin * leverage # Tamaño real de la posición (Notional)
            
            # Pagamos comisión de entrada (sobre el tamaño apalancado)
            entry_fee = position_size * fee_pct
            capital -= entry_fee
            
            entry_price = current_close
            tp_price = entry_price * (1 + TP_PCT)
            sl_price = entry_price * (1 - SL_PCT)
            
            # Registrar entrada
            trade = {
                'entry_time': timestamp,
                'entry_price': entry_price,
                'type': 'LONG',
                'size': position_size,
                'fee_entry': entry_fee
            }
            
            in_position = True
            
            # --- LÓGICA DE SALIDA (LOOK-AHEAD) ---
            # Miramos hacia el futuro para ver cuándo cerramos
            for j in range(1, MAX_HOLD_CANDLES + 1):
                future_idx = i + j
                high = df['high'].iloc[future_idx]
                low = df['low'].iloc[future_idx]
                close_future = df['close'].iloc[future_idx]
                time_future = df.index[future_idx]
                
                exit_price = None
                reason = None
                
                # REGLA CONSERVADORA: Si high > TP y low < SL en la misma vela, asumimos SL.
                
                # 1. Chequeo de Stop Loss
                if low <= sl_price:
                    exit_price = sl_price
                    reason = 'Stop Loss'
                
                # 2. Chequeo de Take Profit (Solo si no tocó SL antes)
                elif high >= tp_price:
                    exit_price = tp_price
                    reason = 'Take Profit'
                    
                # 3. Chequeo de Cierre por Tiempo (Time Stop)
                elif j == MAX_HOLD_CANDLES:
                    exit_price = close_future
                    reason = 'Time Stop'
                
                # Si ocurrió una salida:
                if exit_price:
                    # Cálculo de Ganancia/Pérdida (PnL)
                    # PnL = (Precio Salida - Precio Entrada) / Precio Entrada * Tamaño Posición
                    pnl_pct = (exit_price - entry_price) / entry_price
                    pnl_usdt = pnl_pct * position_size
                    
                    # Comisión de salida
                    exit_val = position_size * (1 + pnl_pct) # Valor final de la posición
                    exit_fee = exit_val * fee_pct
                    
                    # Actualizar Capital
                    capital += pnl_usdt - exit_fee
                    
                    # Guardar datos del trade
                    trade['exit_time'] = time_future
                    trade['exit_price'] = exit_price
                    trade['pnl'] = pnl_usdt
                    trade['net_profit'] = pnl_usdt - entry_fee - exit_fee
                    trade['reason'] = reason
                    trades.append(trade)
                    
                    in_position = False
                    # Saltamos el bucle principal hasta donde terminó el trade para no solapar
                    # (Esto es una simplificación, en código real usaríamos un puntero 'skip')
                    break 
        
        equity_curve.append(capital)

    # 3. RESULTADOS Y VISUALIZACIÓN
    results_df = pd.DataFrame(trades)
    
    if len(results_df) == 0:
        print("No se realizaron operaciones con este Threshold.")
        return
    
    # Cálculo de métricas
    total_trades = len(results_df)
    wins = len(results_df[results_df['net_profit'] > 0])
    win_rate = wins / total_trades
    net_profit = capital - initial_capital
    retorno_neto = ((capital / initial_capital) - 1)*100
    max_drawdown = (pd.Series(equity_curve).cummax() - pd.Series(equity_curve)).max()
    
    print("\n" + "="*40)
    print(f" RESULTADOS BACKTEST (Threshold: {threshold})")
    print("="*40)
    print(f"Capital Inicial:   ${initial_capital:.2f}")
    print(f"Capital Final:     ${capital:.2f}")
    print(f"Ganancia Neta:     ${net_profit:.2f}")
    print(f"Retorno Neto:      {((capital/initial_capital)-1)*100:.2f}%")
    print(f"Total Operaciones: {total_trades}")
    print(f"Win Rate Real:     {win_rate:.2%}")
    print(f"Max Drawdown:      ${max_drawdown:.2f}")
    print("-" * 40)
    print("Desglose por tipo de cierre:")
    print(results_df['reason'].value_counts())
    
    # Gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='Capital (Equity Curve)')
    plt.title(f'Crecimiento de Cuenta (Lev: {leverage}x | TP: {TP_PCT*100}% | SL: {SL_PCT*100}%)')
    plt.xlabel('Velas')
    plt.ylabel('USDT')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results_df

# --- MODO DE USO ---
# Asumiendo que ya tienes:
# 1. df_test (tus datos de prueba con OHLC)
# 2. model (tu XGBoost entrenado)
# 3. feature_cols (la lista de nombres de columnas que usaste para entrenar)

# Ejecuta el backtest:
df_test=data("datos_btc_1año.csv")
df_test=add_indicators(df_test)
model = obtener_modelo_entrenado()
feature_cols = ["day","hour","volume"]
historial_trades = run_backtest(df_test, model, feature_cols, leverage=5)