{-# LANGUAGE FlexibleContexts #-}

module Numeric.Trainee.Data.Common (
	CsvM, runCsvM, col_, read_, single_, enum_, class_, sample_
	) where

import Prelude.Unicode

import Control.DeepSeq
import Control.Monad.Except
import Control.Monad.Reader
import Data.List
import qualified Data.Vector as V
import qualified Data.Text as T
import Numeric.LinearAlgebra
import Text.Read

import Numeric.Trainee.Types

type CsvM a = ReaderT (V.Vector T.Text) (Either String) a

runCsvM ∷ CsvM a → V.Vector T.Text → Either String a
runCsvM = runReaderT

col_ ∷ Int → CsvM String
col_ idx = ReaderT $ \r → maybe (throwError $ "invalid column index: " ++ show idx) (return ∘ T.unpack) (r V.!? idx)

read_ ∷ Read a ⇒ String → CsvM a
read_ s = maybe (throwError $ "can't parse: " ++ s) return ∘ readMaybe $ s

single_ ∷ a → CsvM [a]
single_ = return ∘ return

enum_ ∷ Fractional a ⇒ [String] → String → CsvM a
enum_ names name = case findIndex (≡ name) names of
	Nothing → throwError $ "invalid enum value: " ++ name ++ ", expected " ++ intercalate ", " names
	Just idx → return (fromIntegral idx / fromIntegral (length names - 1))

class_ ∷ Num a ⇒ [String] → String → CsvM [a]
class_ names name = case findIndex (≡ name) names of
	Nothing → throwError $ "invalid class value: " ++ name ++ ", expected " ++ intercalate ", " names
	Just idx → return $ replicate idx 0 ++ [1] ++ replicate (length names - idx - 1) 0

sample_ ∷ Container Vector a ⇒ [a] → [a] → CsvM (Sample (Vector a) (Vector a))
sample_ is os = return $!! Sample (fromList is) (fromList os)
