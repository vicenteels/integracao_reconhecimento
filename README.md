# Integração de reconhecimento facial

`python main.py reconhecer` abre a câmera e a janela local do OpenCV, como no fluxo original. Pressione `q` para sair. Nenhuma página é aberta.

O reconhecimento lê `id`, `nome` e os cinco descritores `embedding_fotos` diretamente de `facial_recognition.professores`, em transações somente leitura; apenas o status em `reservas` é alterado. A câmera é capturada pelo OpenCV. Um processo local do Chrome ou Edge, sem janela, usa os mesmos modelos `@vladmandic/face-api` do cadastro web (copiados em `modelo/face-api`) para gerar descritores compatíveis. A menor distância euclidiana abaixo de 0,6 identifica o professor, seguindo a regra em `../facial-recognition/lib/teachers.ts`.

## Liberação da sala

`ID_SALA` (UUID de `salas.id`, em `.env.local`) define a sala controlada por esta câmera. Quando um professor é reconhecido, a tabela `reservas` dessa sala é consultada:

- reserva `RESERVA_ATIVA` do próprio professor, com `data_hora_inicio <= agora < data_hora_reserva` → passa para `EM_USO` ("Sala liberada");
- reserva `EM_USO` do próprio professor no horário → nada muda ("Sala já liberada"); o reconhecimento nunca finaliza reservas;
- nenhuma reserva vigente, reserva de outro professor ou rosto desconhecido → "Acesso facial negado".

Após cada tentativa o mesmo professor aguarda 15 s antes de nova consulta ao banco. A consulta e a atualização ocorrem em uma única transação com `FOR UPDATE`. A credencial precisa de `SELECT` em `professores` e `SELECT`/`UPDATE` em `reservas`.

## Iniciar

1. Instale o Google Chrome, Chromium ou Edge (o `face-api.js` e os modelos já estão em `modelo/face-api`).
2. Configure `DATABASE_URL` e `ID_SALA` em `.env.local` neste diretório (veja "Liberação da sala" para as permissões).
3. Execute `./.venv/Scripts/python.exe main.py reconhecer` no PowerShell. Use `--camera 1` se a câmera principal tiver outro índice.

O primeiro quadro pode levar alguns segundos enquanto os modelos aquecem. As referências são sincronizadas a cada 30 segundos. Se o banco falhar depois de uma sincronização bem-sucedida, as referências já carregadas permanecem em memória. Os quadros da câmera e os descritores circulam apenas por um servidor temporário em `127.0.0.1`; ele é encerrado junto com a janela. Nenhum dado é gravado no banco ou em logs.

O comando `main.py cadastrar` e o fluxo SFace são legados; não são necessários para usar os embeddings do cadastro web. A tabela `professor_sface_referencias` não é necessária.
