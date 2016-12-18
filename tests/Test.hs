module Main (
	main
	) where

import Control.Concurrent
import Control.Monad
import Control.Monad.State

import Numeric.Trainee.Neural
import Numeric.LinearAlgebra

test ∷ IO (Net Double)
test = net $ input 2 ⭃ fc sigma 2 ⭃ fc sigma 2 ⭃ fc sigma 1

(⇢) ∷ a → b → Sample a b
(⇢) = (,)

samples ∷ [(Vector Double, Vector Double)]
samples = [
	vector [0, 0] ⇢ vector [0],
	vector [1, 1] ⇢ vector [0],
	vector [1, 0] ⇢ vector [1],
	vector [0, 1] ⇢ vector [1]]

main ∷ IO ()
main = do
	n ← test
	(e, n') ← runLearnT n $ trainUntil 1.0 10000 4 0.0001 squared samples
	liftIO $ putStrLn $ "error: " ++ show e
	liftIO $ putStrLn "result:"
	liftIO $ forM_ samples $ \(x, _) →
		putStrLn $ show x ++ " -> " ++ show (eval n' x)
