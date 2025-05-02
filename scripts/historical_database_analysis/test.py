    def retrieve_passages_for_query(self, query: str, top_k: int = 5) -> List[Dict]:
        """为历史query检索相关段落"""
        try:
            logging.info(f"Retrieving passages for query: {query[:50]}...")
            
            # 使用检索数据库中的数据
            query_embedding = self.model.encode([query])[0]
            scored_passages = []
            
            # 批处理处理段落
            batch_size = 32
            for i in range(0, len(self.retrieval_database), batch_size):
                batch = self.retrieval_database[i:i+batch_size]
                texts = [p['text'] for p in batch]
                passage_embeddings = self.model.encode(texts)
                
                for passage, embedding in zip(batch, passage_embeddings):
                    similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                    if similarity > self.retrieval_config['similarity_threshold']:
                        scored_passages.append({
                            'text': passage['text'],
                            'score': float(similarity),
                            'id': passage.get('id', ''),
                            'title': passage.get('title', '')
                        })
                
                if i % 100 == 0:
                    logging.info(f"Processed {i}/{len(self.retrieval_database)} passages")
            
            # 按相关度排序并返回top-k个段落
            scored_passages.sort(key=lambda x: x['score'], reverse=True)
            return scored_passages[:top_k]
                
        except Exception as e:
            logging.error(f"Error retrieving passages for query {query}: {str(e)}")
            return []