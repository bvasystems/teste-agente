import csv
import urllib.request
import io

def consultarPlanilhaVacinas(nome_vacina: str) -> str:
    """
    Consulta as informações ATUAIS de uma vacina na planilha oficial de preços e estoque da clínica.
    Sempre chame esta ferramenta para saber o preço real, disponibilidade atual, quantidade de doses e para que serve.
    
    Args:
        nome_vacina: O nome da vacina pesquisada pelo cliente (ex: 'Hexavalente', 'Meningo B', 'Dengue', 'Pneumo').
        
    Returns:
        Um texto com os detalhes reais da vacina encontrados na planilha oficial da clínica.
    """
    # Link direto para exportar a sua planilha como CSV para leitura do Python
    url = "https://docs.google.com/spreadsheets/d/118fKTq6s_a-9lUx-_PabeuaUafFz9YduQIz9SGFXFUY/export?format=csv"
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            conteudo_csv = response.read().decode('utf-8')
            
        # Parse CSV lendo a primeira linha como cabeçalho
        reader = csv.DictReader(io.StringIO(conteudo_csv))
        
        resultados = []
        termo_busca = nome_vacina.lower().strip()
        
        for row in reader:
            nome_na_planilha = row.get("Vacina", "").strip()
            
            # Se a linha estiver em branco, pula
            if not nome_na_planilha:
                continue
                
            # Verifica se o termo pesquisado faz parte do nome da vacina na planilha
            if termo_busca in nome_na_planilha.lower():
                
                # Extraindo exatamente os nomes das colunas que vimos no teste CUrl
                grupo = row.get("Grupo", "").strip()
                idade = row.get("Doses / Idade", "").strip()
                sobre = row.get("Sobre a Vacina", "").strip()
                esquema = row.get("Esquema Vacinal", "").strip()
                preco = row.get("Preço (R$) / dose", "").strip()
                em_falta = row.get("FALTA EM ESTOQUE", "FALSE").strip().upper()
                
                status_estoque = "🔴 EM FALTA (Sem previsão)" if em_falta == "TRUE" else "🟢 Disponível no momento"
                
                detalhes = f"💉 Vacina: {nome_na_planilha}\n"
                if grupo: detalhes += f"  👥 Atende: {grupo}\n"
                if idade: detalhes += f"  📅 Indicação: {idade}\n"
                if sobre: detalhes += f"  ℹ️ Para que serve: {sobre}\n"
                if esquema: detalhes += f"  🔄 Esquema vacinal: {esquema}\n"
                
                detalhes += f"  📦 Estoque: {status_estoque}\n"
                
                # Se tiver em falta, avisar explicitamente.
                if em_falta == "TRUE":
                    detalhes += "  💰 Valor: (Indisponível para venda pois está em falta)\n"
                elif preco: 
                    detalhes += f"  💰 Valor: {preco}\n"
                else:
                    detalhes += "  💰 Valor: (Preço não preenchido na planilha)\n"
                
                resultados.append(detalhes)
        
        if not resultados:
            return f"Não encontrei nenhuma vacina correspondente a '{nome_vacina}' na planilha atual."
            
        return "Resultados direto da planilha oficial:\n\n" + "\n\n".join(resultados)
        
    except Exception as e:
        return f"Erro de sistema ao tentar se conectar com o Google Sheets: {str(e)}"
